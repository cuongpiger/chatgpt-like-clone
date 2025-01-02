from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from gradio import ChatMessage as GradioChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

import gradio as gr
from typing import List
import sys

raw_document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

pipeline = Pipeline()
pipeline.add_component("converter", PDFMinerToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=20, split_overlap=15))
pipeline.add_component("writer", DocumentWriter(document_store=raw_document_store))

pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "writer")

file_names = ["sample.pdf"]
pipeline.run({"converter": {"sources": file_names}})

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(raw_document_store.filter_documents())
document_store.write_documents(docs_with_embeddings["documents"])

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
# Create generator component
model_name = "llama3.1:8b"
chat_generator = OllamaChatGenerator(
    model=model_name,  # llama3.3 or llama3.1:8b
    streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),
    url=sys.argv[1],
    generation_kwargs={
        "num_predict": 1024,
        "temperature": 0.01})

template = [ChatMessage.from_user("""
Given the following information, answer the question.

{% if memories is defined and memories | length > 0 %}
Conversation history:
{% for memory in memories %}
    {{ memory.text }}
{% endfor %}
{% endif %}

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")]

prompt_builder = ChatPromptBuilder(template=template)
retriever = InMemoryEmbeddingRetriever(document_store)
basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)
# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

chat_history: List[GradioChatMessage] = []


def chat(message, history):
    result = basic_rag_pipeline.run(
        {"text_embedder": {"text": message},
         "prompt_builder": {"question": message, "memories": chat_history[-10:]}})
    llm_response = result["llm"]["replies"][0]
    print(f"llm_response: {llm_response}")
    chat_history.append(GradioChatMessage(role="user", content=message))
    chat_history.append(GradioChatMessage(role="assistant", content=llm_response.content))

    return llm_response.content


questions = [
    "What is VKS?",
    "Compare VKS private clusters and VKS public clusters",
    "What is vngcloud-blockstorage-csi-driver?",
    "I want to create a pvc of 20Gi using vngcloud-blockstorage-csi-driver",
]

demo = gr.ChatInterface(
    fn=chat,
    examples=questions,
    title="ChatGPT-like clone",
)
demo.launch(server_name="0.0.0.0", share=True)

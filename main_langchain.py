import gradio as gr
from typing import List
import sys

import os

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PGVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing_extensions import List, TypedDict, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing_extensions import List, TypedDict, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "/home/stackops/secret/work/vngcloud/ai-platform/vertex-ai-credential.json"
)

llm = ChatVertexAI(model="gemini-1.5-flash")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"
vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)
retriever = vectorstore.as_retriever()

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

config = {
        "configurable": {"session_id": "abc123"}
    }

def chat(message, history):
    res = conversational_rag_chain.invoke(
        {"input": message},
        stream_mode="values",
        config=config,
    )

    print(res)
    llm_response = res["answer"]
    return llm_response


questions = [
    "So sánh VKS private clusters và VKS public clusters.",
    "Tôi muốn tạo một PVC 20Gi bằng vngcloud-blockstorage-csi-driver",
    "VKS có những tính năng gì nổi bật?",
    "VKS hỗ trợ các Kubernetes version nào?"
]

if __name__ == '__main__':
    demo = gr.ChatInterface(
        fn=chat,
        examples=questions,
        title="ChatGPT-like clone",
    )
    demo.launch(server_name="0.0.0.0", share=True)

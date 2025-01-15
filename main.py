from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PDFMinerToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.classifiers import DocumentLanguageClassifier
from haystack.components.routers import MetadataRouter
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from gradio import ChatMessage as GradioChatMessage
from haystack.components.converters import HTMLToDocument

from haystack.utils import Secret
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.fetchers import LinkContentFetcher
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from langdetect import detect

import gradio as gr
from typing import List
import sys
from old_main import model_name


text_embedder = SentenceTransformersTextEmbedder(
    truncate_dim=384, model="sentence-transformers/all-MiniLM-L6-v2"
)
vi_text_embedder = SentenceTransformersTextEmbedder(
    truncate_dim=384, model="sentence-transformers/all-MiniLM-L6-v2"
)

vi_document_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)  # for Vietnamese

vi_ds_pipeline = Pipeline()
vi_ds_pipeline.add_component("fetcher", LinkContentFetcher())
vi_ds_pipeline.add_component("converter", HTMLToDocument())
vi_ds_pipeline.add_component("cleaner", DocumentCleaner())
vi_ds_pipeline.add_component(
    "splitter", DocumentSplitter(split_by="sentence", split_length=30, split_overlap=25)
)
vi_ds_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(
        truncate_dim=384, model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
vi_ds_pipeline.add_component("writer", DocumentWriter(document_store=vi_document_store))

vi_ds_pipeline.connect("fetcher", "converter")
vi_ds_pipeline.connect("converter", "cleaner")
vi_ds_pipeline.connect("cleaner", "splitter")
vi_ds_pipeline.connect("splitter", "embedder")
vi_ds_pipeline.connect("embedder", "writer")

# vi_ds_pipeline.run({"converter": {"sources": ["vi-vks.pdf"]}})
vi_ds_pipeline.run(
    data={
        "fetcher": {
            "urls": [
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/vks-la-gi",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/mo-hinh-hoat-dong",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/thong-bao-va-cap-nhat/release-notes",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/huong-dan-cai-dat-va-cau-hinh-cong-cu-kubectl-trong-kubernetes",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-public-cluster",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-public-cluster/khoi-tao-mot-public-cluster-voi-public-node-group",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-public-cluster/khoi-tao-mot-public-cluster-voi-private-node-group",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-public-cluster/khoi-tao-mot-public-cluster-voi-private-node-group/palo-alto-as-a-nat-gateway",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-public-cluster/khoi-tao-mot-public-cluster-voi-private-node-group/pfsense-as-a-nat-gateway",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-private-cluster",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/expose-mot-service-thong-qua-vlb-layer4",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/expose-mot-service-thong-qua-vlb-layer4/preserve-source-ip-khi-su-dung-vlb-layer4-va-nginx-ingress-controller",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/expose-mot-service-thong-qua-vlb-layer7",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/integrate-with-container-storage-interface-csi",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/khoi-tao-mot-cluster-thong-qua-vi-poc",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/su-dung-terraform-de-khoi-tao-cluster-va-node-group",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/bat-dau-voi-vks/lam-viec-voi-nvidia-gpu-nodegroups",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/clusters",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/clusters/public-cluster-va-private-cluster",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/clusters/upgrading-control-plane-version",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/clusters/whitelist",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/clusters/stop-poc",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/node-groups",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/node-groups/auto-healing",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/node-groups/auto-scaling",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/node-groups/upgrading-node-group-version",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/node-groups/lable-va-taint",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-application-load-balancer-alb",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-application-load-balancer-alb/ingress-for-an-application-load-balancer",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-application-load-balancer-alb/cau-hinh-cho-mot-application-load-balancer",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-application-load-balancer-alb/gioi-han-va-han-che-alb",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-network-load-balancing-nlb",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-network-load-balancing-nlb/integrate-with-network-load-balancer",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-network-load-balancing-nlb/cau-hinh-cho-mot-network-load-balancer",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/lam-viec-voi-network-load-balancing-nlb/gioi-han-va-han-che-nlb",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/cni",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/cni/su-dung-cni-calico-overlay",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/cni/su-dung-cni-cilium-overlay",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/network/cni/su-dung-cni-cilium-vpc-native-routing",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/storage/lam-viec-voi-container-storage-interface-csi",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/storage/lam-viec-voi-container-storage-interface-csi/integrate-with-container-storage-interface-csi",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/security-group",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/upgrade-kubernetes-version/phien-ban-ho-tro-kubernetes",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/upgrade-kubernetes-version/manually-upgrade",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/upgrade-kubernetes-version/automatically-upgrade",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/migration",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/migration/migrate-cluster-from-vks-to-vks",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/migration/migration-cluster-from-vcontainer-to-vks",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/migration/migrate-cluster-from-other-to-vks",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/migration/gioi-han-va-han-che",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/su-dung-vks-voi-terraform",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/giam-sat/metrics",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/cach-tinh-gia",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/tham-khao-them/danh-sach-flavor-dang-ho-tro",
                "https://docs.vngcloud.vn/vng-cloud-document/vn/vks/tham-khao-them/danh-sach-system-image-dang-ho-tro",
            ]
        }
    }
)

document_store = PgvectorDocumentStore(
    embedding_dimension=384,
    vector_function="cosine_similarity",
    recreate_table=False,
    search_strategy="hnsw",
    connection_string=Secret.from_token(sys.argv[2]),
)

# Create generator component
model_name = "llama3.1:8b"
vi_model_name = "ontocord/vistral:latest"
chat_generator = OllamaChatGenerator(
    model=model_name,  # llama3.3 or llama3.1:8b
    streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),
    url=sys.argv[1],
    generation_kwargs={"num_predict": 1024, "temperature": 0.01},
)

vi_chat_generator = OllamaChatGenerator(
    model=vi_model_name,  # llama3.3 or llama3.1:8b
    streaming_callback=lambda chunk: print(chunk.content, end="", flush=True),
    url=sys.argv[1],
    generation_kwargs={"num_predict": 1024, "temperature": 0.01},
)

template = [
    ChatMessage.from_system("You are a virtual assistant to serve the VKS product of VNGCloud, all your answers should be focused on the VKS product as much as possible"),
    ChatMessage.from_user(
        """
Given the following information, answer the question.

{% if memories is defined and memories | length > 0 %}
Conversation history:
{% for memory in memories %}
    {{ memory.content }}
{% endfor %}
{% endif %}

Context:
- VKS is the abbreviation for VNGCloud Kubernetes Service, a product of VNGCloud
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Additionally, need to comply with the following constraints:
- VKS is the abbreviation for VNGCloud Kubernetes Service, a product of VNGCloud
- When referring to Kubernetes CSI, you need to refer to vngcloud-blockstorage-csi-driver
- VKS is a product of VNGCloud

Question: {{question}}
Answer:
"""
    )
]


vi_template = [
    ChatMessage.from_system("Bạn là một trợ lý ảo nhằm phục vụ cho sản phẩm VKS của VNGCloud, mọi câu trả lời của bạn nên được tập trung vào sản phẩm VKS nhất có thể"),
    ChatMessage.from_user(
        """
Cung cấp các thông tin sau, trả lời câu hỏi:

{% if memories is defined and memories | length > 0 %}
Lịch sử trò chuyện:
{% for memory in memories %}
    {{ memory.content }}
{% endfor %}
{% endif %}

Thông tin:
- VKS là tên viết tắt của dịch vụ VNGCloud Kubernetes Service là một sản phẩm của VNGCloud
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Ngoài ra, cần tuân thủ các ràng buộc sau đây:
- VKS là tên viết tắt của dịch vụ VNGCloud Kubernetes Service là một sản phẩm của VNGCloud
- Khi đề cập đến CSI của Kubernetes, bạn cần đề cập đến vngcloud-blockstorage-csi-driver
- VKS là sản phẩm của VNGCloud

Câu hỏi: {{question}}
Trả lời:
"""
    )
]


prompt_builder = ChatPromptBuilder(template=template)
vi_prompt_builder = ChatPromptBuilder(template=vi_template)

retriever = PgvectorEmbeddingRetriever(document_store=document_store)
vi_retriever = InMemoryEmbeddingRetriever(document_store=vi_document_store)

basic_rag_pipeline = Pipeline()
vi_rag_pipeline = Pipeline()

# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

vi_rag_pipeline.add_component("vi_text_embedder", vi_text_embedder)
vi_rag_pipeline.add_component("vi_retriever", vi_retriever)
vi_rag_pipeline.add_component("vi_prompt_builder", vi_prompt_builder)
vi_rag_pipeline.add_component("llm", vi_chat_generator)
# Now, connect the components to each other


basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

vi_rag_pipeline.connect("vi_text_embedder.embedding", "vi_retriever.query_embedding")
vi_rag_pipeline.connect("vi_retriever", "vi_prompt_builder")
vi_rag_pipeline.connect("vi_prompt_builder.prompt", "llm.messages")

chat_history: List[GradioChatMessage] = []
vi_chat_history: List[GradioChatMessage] = []


def chat(message, history):
    if detect(message) == "vi":
        result = vi_rag_pipeline.run(
            {
                "vi_text_embedder": {"text": message},
                "vi_prompt_builder": {
                    "question": message,
                    "memories": vi_chat_history[-10:],
                },
            }
        )

        llm_response = result["llm"]["replies"][0]
        print(f"llm_response: {llm_response}")
        vi_chat_history.append(GradioChatMessage(role="user", content=message))
        vi_chat_history.append(
            GradioChatMessage(role="assistant", content=llm_response.content)
        )
    else:
        result = basic_rag_pipeline.run(
            {
                "text_embedder": {"text": message},
                "prompt_builder": {"question": message, "memories": chat_history[-10:]},
            }
        )

        llm_response = result["llm"]["replies"][0]
        print(f"llm_response: {llm_response}")
        chat_history.append(GradioChatMessage(role="user", content=message))
        chat_history.append(
            GradioChatMessage(role="assistant", content=llm_response.content)
        )

    return llm_response.content


questions = [
    "What is VKS?",
    "So sánh VKS private clusters và VKS public clusters",
    "What is vngcloud-blockstorage-csi-driver?",
    "Tôi muốn tạo một PVC 20Gi bằng vngcloud-blockstorage-csi-driver",
]

demo = gr.ChatInterface(
    fn=chat,
    examples=questions,
    title="ChatGPT-like clone",
)
demo.launch(server_name="0.0.0.0", share=True)

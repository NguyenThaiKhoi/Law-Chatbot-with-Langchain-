from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Loading the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/LaBSE"
)

# Loading the documents
loader = DirectoryLoader("./data/", glob="*.txt")
documents = loader.load()

for doc in documents:
    doc.page_content = doc.page_content.replace("\n", " ")

# Splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
docs = text_splitter.split_documents(documents)

client = QdrantClient(
    url="https://f8ce82b8-6f70-4a47-80ff-b572cce2c43f.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="zQdpYjaq4vVDdiOgdzDc2MJt9nEnd_OzykYrFdLbUyyK22VZu_Wa1A"
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="Chatbot_law",
    embedding=embeddings
)

print("Documents Vectorized")
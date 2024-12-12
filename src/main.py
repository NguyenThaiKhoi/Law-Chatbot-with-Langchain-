import os 
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq


working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def setup_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE"
    )
    client = QdrantClient(
        url="https://f8ce82b8-6f70-4a47-80ff-b572cce2c43f.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key="zQdpYjaq4vVDdiOgdzDc2MJt9nEnd_OzykYrFdLbUyyK22VZu_Wa1A",
        timeout=60  # Increase timeout to 60 seconds
    )

    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="Chatbot_law",
            embedding=embeddings
        )
        # Test the connection
        client.get_collections()
        return vector_store
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {str(e)}")
        return None



def chat_chain(vector_store):
    system_template = """Bạn là một chuyên gia tư vấn trong lĩnh vực luật giao thông, tên của bạn là Pika. 
    Bạn có nhiệm vụ cung cấp thông tin chính xác và dễ hiểu về các quy định, điều luật, và hướng dẫn liên quan đến giao thông đường bộ tại Việt Nam. 
    Trả lời câu hỏi của người dùng bằng ngôn ngữ thân thiện, ngắn gọn, và dễ hiểu, nhưng vẫn đảm bảo tính chính xác pháp lý. 
    Nếu cần thiết, hãy cung cấp ví dụ minh họa thực tế và giải thích các thuật ngữ phức tạp. 
    Khi không thể trả lời hoặc thông tin không rõ ràng, hãy khuyến nghị người dùng tham khảo các nguồn thông tin chính thống hoặc liên hệ cơ quan có thẩm quyền.

    Dựa trên thông tin sau đây để trả lời câu hỏi của người dùng:
    {context}

    Câu hỏi: {question}
    Trả lời hữu ích:"""

    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(model="llama3-8b-8192", temperature=0, stop_sequences=["\n\nHuman: "])
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

st.set_page_config(
    page_title="PikaGPT",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("PikaGPT")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = setup_vector_store()
    
if st.session_state.vector_store is None:
    st.error("Could not connect to the vector store. Please try again later.")
    st.stop()

if "conversation_chain" not in st.session_state:   
    st.session_state.conversation_chain = chat_chain(st.session_state.vector_store)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Tôi có thể giúp gì cho bạn?")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

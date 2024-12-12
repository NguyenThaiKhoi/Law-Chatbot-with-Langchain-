# Pika Chatbot - Vietnamese Traffic Law Assistant

A specialized chatbot built to provide accurate and helpful information about Vietnamese traffic laws and regulations. Named "Pika," this AI assistant aims to make legal information more accessible and understandable to the general public.

## Features

- 🤖 Interactive AI assistant specialized in Vietnamese traffic laws
- 💬 Natural language understanding and response in Vietnamese
- 📚 Access to comprehensive traffic law information
- 🔍 Context-aware responses with source references
- 🌐 Web-based interface using Streamlit

## Prerequisites

- Python 3.9+
- Groq API key
- Qdrant vector database access

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd Pika_Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your environment:
   - Create a `config.json` file with your API keys:
   ```json
   {
       "GROQ_API_KEY": "your-groq-api-key"
   }
   ```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Start chatting with Pika about Vietnamese traffic laws!

## Project Structure

- `main.py` - Main application file containing the Streamlit interface and chat logic
- `vectorize_documents.py` - Document processing and vectorization utilities
- `requirements.txt` - Project dependencies with specific versions
- `config.json` - Configuration file for API keys

## Features

- Natural language interaction in Vietnamese
- Context-aware responses based on official traffic law documents
- User-friendly web interface
- Memory of conversation context
- Source citations for legal information

## Dependencies

Major dependencies include:
- langchain-text-splitters==0.3.2
- langchain-community==0.3.11
- langchain-groq==0.2.1
- langchain-huggingface==0.1.2
- streamlit==1.41.0
- qdrant-client==1.12.1
- sentence-transformers==3.3.1

For a complete list, see `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Groq](https://groq.com/)
- Vector storage by [Qdrant](https://qdrant.tech/)

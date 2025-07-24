import os
from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Chemin vers tes documents
DOCUMENTS_DIR = "documents/"

# Fonction pour charger tous les documents
def load_documents():
    docs = []
    for filename in os.listdir(DOCUMENTS_DIR):
        path = os.path.join(DOCUMENTS_DIR, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Préparer les documents
def create_vector_store():
    documents = load_documents()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    return vectorstore

vectorstore = create_vector_store()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        query = request.form["query"]
        answer = qa.run(query)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

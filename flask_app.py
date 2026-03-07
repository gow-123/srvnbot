import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.messages import HumanMessage

load_dotenv()

app = Flask(__name__)

CV_DIR = "CV"
VECTOR_STORE_PATH = "faiss_index"
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

print("Initializing models...")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if Path(VECTOR_STORE_PATH).exists():
    vectors = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    loader = PyPDFDirectoryLoader(CV_DIR)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_docs = splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_docs, embeddings)
    vectors.save_local(VECTOR_STORE_PATH)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=LLM_MODEL,
    temperature=0,
)

retriever = vectors.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are Saravana Perumal R. — a Data Scientist and AI practitioner.
Answer in first person.

Context:
{context}

Question: {question}

Answer:
""")

print("System ready.")



@app.route("/")
def home():
    return send_from_directory('.', 'index.html')




@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return jsonify({"status": "Send a POST request with JSON body: {message: your question}"})

    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    user_query = data.get("message")

    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    # FIXED: use .invoke() instead of .get_relevant_documents()
    docs = retriever.invoke(user_query)

    if not docs:
        return jsonify({"response": "No relevant information found."})

    context = "\n\n".join([doc.page_content for doc in docs])

    formatted_prompt = prompt.format(context=context, question=user_query)

    response = llm.invoke([HumanMessage(content=formatted_prompt)])

    return jsonify({"response": response.content})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
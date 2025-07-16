# create_index.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# کلید API خود را اینجا قرار دهید (یا از متغیرهای محیطی بخوانید)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
PDF_PATH = "company_knowledge.pdf"
FAISS_INDEX_PATH = "faiss_index" # نام پوشه برای ذخیره ایندکس

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print("Creating embeddings and FAISS index... (This may take a while)")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
vector_store = FAISS.from_documents(chunks, embeddings_model)

print(f"Saving FAISS index to '{FAISS_INDEX_PATH}'...")
vector_store.save_local(FAISS_INDEX_PATH)

print("✅ Index created and saved successfully!")
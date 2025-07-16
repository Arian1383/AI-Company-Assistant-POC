# create_index.py
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- مرحله ۱: کلید API جدید خود را اینجا قرار دهید ---
# این کلید را پس از حذف کلید قبلی و ساختن یک کلید جدید، در اینجا کپی کنید.
# هرگز این فایل را با کلید واقعی در گیت‌هاب عمومی قرار ندهید.
GOOGLE_API_KEY = "AIzaSyBzAKJqZKdQQmkuVbPoA2AvLGqZ7uXjUhI" 

# --- تنظیمات فایل‌ها ---
PDF_PATH = "company_knowledge.pdf"
FAISS_INDEX_PATH = "faiss_index" 

# --- بررسی اولیه ---
if GOOGLE_API_KEY == "AIzaSyBzAKJqZKdQQmkuVbPoA2AvLGqZ7uXjUhI":
    print("🚨 خطا: لطفاً قبل از اجرا، کلید API جدید خود را در متغیر GOOGLE_API_KEY قرار دهید.")
    exit()

if not os.path.exists(PDF_PATH):
    print(f"🚨 خطا: فایل '{PDF_PATH}' پیدا نشد. لطفاً مطمئن شوید این فایل در کنار اسکریپت قرار دارد.")
    exit()

# --- فرآیند ساخت ایندکس ---
try:
    print("1. در حال خواندن فایل PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("2. در حال تقسیم‌بندی متن به قطعات کوچکتر...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print("3. در حال ساخت Embeddings و ایندکس FAISS... (این مرحله ممکن است کمی طول بکشد)")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_documents(chunks, embeddings_model)

    print(f"4. در حال ذخیره ایندکس در پوشه '{FAISS_INDEX_PATH}'...")
    vector_store.save_local(FAISS_INDEX_PATH)

    print("\n✅ عملیات با موفقیت انجام شد! پوشه 'faiss_index' ساخته شد.")
    print("حالا می‌توانید این پوشه را به همراه سایر فایل‌ها در گیت‌هاب آپلود کرده و برنامه را دیپلوی کنید.")

except Exception as e:
    print(f"\n❌ یک خطای غیرمنتظره رخ داد: {e}")
    print("لطفاً موارد زیر را بررسی کنید:")
    print("- از اتصال اینترنت خود مطمئن شوید.")
    print("- مطمئن شوید کلید API شما صحیح و فعال است و محدودیت ندارد.")
    print("- مطمئن شوید حساب پرداخت (Billing) به پروژه گوگل کلاد شما متصل است.")


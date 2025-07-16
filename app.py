import streamlit as st
import os
import json
import logging
import uuid
import tempfile
import base64
from contextlib import contextmanager
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Configure Logging ---
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

# --- Global App ID ---
app_id = str(uuid.uuid4())

# --- Initialize Session State ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'knowledge_vector_store' not in st.session_state:
    st.session_state.knowledge_vector_store = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# --- File Paths ---
USERS_FILE = "users.json"
PDF_FILE_PATH = "company_knowledge.pdf"

# --- Page Configuration ---
st.set_page_config(
    page_title="دستیار دانش شرکت سپاهان",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Theme Management ---
def set_theme(theme_name):
    st.session_state.theme = theme_name
    st.experimental_rerun()

# --- Load CSS ---
def load_css(theme):
    try:
        css_file = "style.css" if theme == "light" else "style_dark.css"
        if os.path.exists(css_file):
            with open(css_file, encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            logger.warning(f"CSS file '{css_file}' not found. Using default Streamlit theme.")
            st.warning(f"⚠️ فایل CSS '{css_file}' پیدا نشد. از تم پیش‌فرض Streamlit استفاده می‌شود.")
    except Exception as e:
        logger.error(f"Error loading CSS file: {e}")
        st.error(f"خطا در بارگذاری فایل CSS: {e}")

load_css(st.session_state.theme)

# --- API Key ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
except KeyError:
    logger.error("Google API Key not found in environment or Streamlit secrets.")
    st.error("🔑 خطای کلید API: کلید Google Gemini پیدا نشد. لطفاً آن را تنظیم کنید.")
    st.stop()

# --- Authentication Functions ---
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        default_users = {
            "users": [{"username": "Sepahan", "password": "Arian", "creation_time": datetime.now().isoformat()}],
            "admin_users": [{"username": "admin_sepahan", "password": "Arian", "creation_time": datetime.now().isoformat()}]
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=4)
        logger.info("Created default users.json")
        return default_users
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        st.error(f"خطا در بارگذاری کاربران: {e}")
        return {"users": [], "admin_users": []}

def save_users(users_data):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users_data, f, indent=4)
        logger.info("Users data saved successfully")
    except Exception as e:
        logger.error(f"Error saving users: {e}")
        st.error(f"خطا در ذخیره کاربران: {e}")

def user_login_local(username, password):
    users_data = load_users()
    for user_info in users_data["users"]:
        if user_info["username"] == username and user_info["password"] == password:
            st.session_state.user_id = username
            st.session_state.user_email = username
            st.session_state.authenticated = True
            st.session_state.is_admin = False
            logger.info(f"User {username} logged in successfully")
            return True
    st.error("❌ نام کاربری یا رمز عبور اشتباه است.")
    logger.warning(f"Failed login attempt for user {username}")
    return False

def admin_login_local(username, password):
    users_data = load_users()
    for admin_info in users_data["admin_users"]:
        if admin_info["username"] == username and admin_info["password"] == password:
            st.session_state.user_id = username
            st.session_state.user_email = username
            st.session_state.authenticated = True
            st.session_state.is_admin = True
            logger.info(f"Admin {username} logged in successfully")
            return True
    st.error("❌ نام کاربری یا رمز عبور مدیر اشتباه است.")
    logger.warning(f"Failed admin login attempt for {username}")
    return False

def logout():
    st.session_state.clear()
    logger.info("User logged out")
    st.experimental_rerun()

# --- User Management Functions ---
def create_user_local(username, password, is_admin_user=False):
    users_data = load_users()
    if any(u["username"] == username for u in users_data["users"]) or \
       any(u["username"] == username for u in users_data["admin_users"]):
        st.warning("⚠️ کاربری با این نام کاربری از قبل وجود دارد.")
        logger.warning(f"Attempt to create duplicate user: {username}")
        return False
    if is_admin_user:
        users_data["admin_users"].append({"username": username, "password": password, "creation_time": datetime.now().isoformat()})
    else:
        users_data["users"].append({"username": username, "password": password, "creation_time": datetime.now().isoformat()})
    save_users(users_data)
    st.success(f"✅ کاربر '{username}' با موفقیت ایجاد شد.")
    logger.info(f"Created user: {username}, is_admin: {is_admin_user}")
    return True

def delete_user_local(username, is_admin_user=False):
    users_data = load_users()
    if is_admin_user:
        initial_len = len(users_data["admin_users"])
        users_data["admin_users"] = [u for u in users_data["admin_users"] if u["username"] != username]
        if len(users_data["admin_users"]) < initial_len:
            save_users(users_data)
            st.success(f"✅ کاربر مدیر '{username}' با موفقیت حذف شد.")
            logger.info(f"Deleted admin user: {username}")
            return True
    else:
        initial_len = len(users_data["users"])
        users_data["users"] = [u for u in users_data["users"] if u["username"] != username]
        if len(users_data["users"]) < initial_len:
            save_users(users_data)
            st.success(f"✅ کاربر عادی '{username}' با موفقیت حذف شد.")
            logger.info(f"Deleted regular user: {username}")
            return True
    st.warning("⚠️ کاربری با این نام کاربری یافت نشد.")
    logger.warning(f"Attempt to delete non-existent user: {username}")
    return False

def list_users_local():
    users_data = load_users()
    all_users = []
    for user in users_data["users"]:
        all_users.append({"username": user["username"], "type": "عادی", "creation_time": user["creation_time"]})
    for admin in users_data["admin_users"]:
        all_users.append({"username": admin["username"], "type": "مدیر", "creation_time": admin["creation_time"]})
    return all_users

# --- PDF Processing Function ---
@contextmanager
def temporary_file(suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file
    finally:
        try:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
                logger.info(f"Temporary file {temp_file.name} deleted")
        except Exception as e:
            logger.error(f"Error deleting temporary file {temp_file.name}: {e}")

def process_pdf_for_rag(pdf_file, source_name):
    try:
        with temporary_file(suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        
        if not documents:
            st.warning(f"⚠️ هیچ متنی از فایل '{source_name}' استخراج نشد.")
            logger.warning(f"No text extracted from PDF: {source_name}")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Processed PDF {source_name} with {len(chunks)} chunks")
        return chunks
    except Exception as e:
        st.error(f"❌ خطایی در پردازش فایل '{source_name}': {e}")
        logger.error(f"Error processing PDF {source_name}: {e}")
        return []

# --- Load Knowledge Base ---
@st.cache_resource(ttl=3600)
def load_knowledge_base_local(api_key, model_name="models/text-embedding-004"):
    all_documents_for_rag = []
    if os.path.exists(PDF_FILE_PATH):
        try:
            loader = PyPDFLoader(PDF_FILE_PATH)
            all_documents_for_rag.extend(loader.load())
            logger.info(f"Loaded PDF: {PDF_FILE_PATH}")
        except Exception as e:
            st.error(f"🚨 خطا در خواندن فایل '{PDF_FILE_PATH}': {e}")
            logger.error(f"Error loading PDF {PDF_FILE_PATH}: {e}")
            return None, []
    else:
        st.error(f"🚨 فایل '{PDF_FILE_PATH}' پیدا نشد.")
        logger.error(f"PDF file {PDF_FILE_PATH} not found")
        return None, []

    if not all_documents_for_rag:
        st.error("🚨 هیچ متنی از فایل استخراج نشد.")
        logger.error(f"No text extracted from PDF {PDF_FILE_PATH}")
        return None, []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents_for_rag)
    embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    logger.info(f"Created vector store with {len(chunks)} chunks")
    return vector_store, chunks

# --- Admin Panel Page ---
def admin_panel_page():
    st.header("🛠️ پنل مدیریت", divider="red")
    st.subheader("مدیریت اسناد و کاربران")

    st.sidebar.title("تنظیمات")
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        set_theme("dark" if st.session_state.theme == "light" else "light")
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    st.warning("⚠️ تغییرات در این نسخه POC دائمی نیستند.")

    # User Management
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>مدیریت کاربران</h4>", unsafe_allow_html=True)
    
    with st.expander("ایجاد کاربر جدید"):
        new_user_username = st.text_input("نام کاربری جدید", key="new_user_username")
        new_user_password = st.text_input("رمز عبور جدید", type="password", key="new_user_password")
        is_new_user_admin = st.checkbox("کاربر مدیر باشد؟", key="is_new_user_admin")
        if st.button("ایجاد کاربر"):
            if new_user_username and new_user_password:
                create_user_local(new_user_username, new_user_password, is_new_user_admin)
            else:
                st.warning("لطفاً نام کاربری و رمز عبور را وارد کنید.")

    with st.expander("حذف کاربر"):
        delete_username = st.text_input("نام کاربری برای حذف", key="delete_username")
        is_delete_user_admin = st.checkbox("کاربر مدیر است؟", key="is_delete_user_admin")
        if st.button("حذف کاربر"):
            if delete_username:
                delete_user_local(delete_username, is_delete_user_admin)
            else:
                st.warning("لطفاً نام کاربری را وارد کنید.")

    st.markdown("<h5 style='text-align: right; color: #f0f0f0; margin-top: 20px; margin-bottom: 15px;'>لیست کاربران:</h5>", unsafe_allow_html=True)
    users_list_data = list_users_local()
    if users_list_data:
        for user_info in users_list_data:
            st.write(f"- **نام کاربری:** {user_info['username']}, **نوع:** {user_info['type']}, **تاریخ ایجاد:** {user_info['creation_time']}")
    else:
        st.info("هیچ کاربری یافت نشد.")

    # Knowledge Base Management
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>مدیریت پایگاه دانش</h4>", unsafe_allow_html=True)
    
    st.info("📚 سند اصلی: 'company_knowledge.pdf'")
    if os.path.exists(PDF_FILE_PATH):
        st.markdown(f"- **نام فایل:** `{os.path.basename(PDF_FILE_PATH)}`")
        st.markdown(f"- **حجم فایل:** `{os.path.getsize(PDF_FILE_PATH) / (1024*1024):.2f} MB`")
        st.markdown("- **وضعیت:** بارگذاری شده.")
    else:
        st.error(f"🚨 فایل '{PDF_FILE_PATH}' یافت نشد.")

# --- User Chat Page ---
def user_chat_page():
    st.sidebar.title("تنظیمات")
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        set_theme("dark" if st.session_state.theme == "light" else "light")
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    st.header("🧠 دستیار دانش هوشمند شرکت سپاهان", divider="red")
    st.subheader("💡 سوالات خود را بپرسید")

    vector_store, all_chunks = load_knowledge_base_local(google_api_key)
    if vector_store is None:
        st.error("🚨 پایگاه دانش بارگذاری نشد.")
        return

    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    st.markdown("<h3 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 10px; margin-top: 40px; margin-bottom: 20px;'>🖼️ افزودن فایل/تصویر</h3>", unsafe_allow_html=True)
    user_uploaded_context_file = st.file_uploader(
        "یک فایل PDF یا تصویر آپلود کنید.",
        type=["pdf", "jpg", "jpeg", "png"],
        key="user_context_uploader"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="تصویر آپلود شده", use_column_width=True)
                st.markdown(message["text_content"])

    if prompt :=
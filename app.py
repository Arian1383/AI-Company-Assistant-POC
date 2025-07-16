import streamlit as st
import os
import json
import tempfile
import base64
from datetime import datetime

# LangChain and AI related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Global App ID (for local file naming) ---
app_id = "sepahan-ai-assistant-local" 

# --- File Paths for Local Storage ---
USERS_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index" # Path to the pre-built FAISS index folder

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="دستیار دانش شرکت سپاهان",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Initialize session state variables safely ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_page = "login"
    st.session_state.authenticated = False
    st.session_state.is_admin = False
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.messages = []
    st.session_state.theme = "light"

# --- Custom CSS Loader ---
def load_css(theme):
    css_file = "style.css" if theme == "light" else "style_dark.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ فایل CSS '{css_file}' پیدا نشد. از تم پیش‌فرض Streamlit استفاده می‌شود.")

# Apply selected theme CSS at the start
load_css(st.session_state.theme)

# --- API Key and Model Settings ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("🔑 خطای کلید API: کلید Google Gemini پیدا نشد. لطفاً آن را در Streamlit Secrets تنظیم کنید.")
    st.stop()

# --- Authentication Functions (Local) ---
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Create default users.json if not exists
    default_users = {
        "users": [
            {"username": "Sepahan", "password": "Arian", "creation_time": datetime.now().isoformat()}
        ],
        "admin_users": [
            {"username": "admin_sepahan", "password": "admin_pass", "creation_time": datetime.now().isoformat()}
        ]
    }
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(default_users, f, indent=4)
    return default_users

def save_users(users_data):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=4)

def user_login_local(username, password):
    users_data = load_users()
    for user_info in users_data["users"]:
        if user_info["username"] == username and user_info["password"] == password:
            st.session_state.user_id = username
            st.session_state.user_email = username
            st.session_state.authenticated = True
            st.session_state.is_admin = False
            return True
    st.error("❌ نام کاربری یا رمز عبور اشتباه است.")
    return False

def admin_login_local(username, password):
    users_data = load_users()
    for admin_info in users_data["admin_users"]:
        if admin_info["username"] == username and admin_info["password"] == password:
            st.session_state.user_id = username
            st.session_state.user_email = username
            st.session_state.authenticated = True
            st.session_state.is_admin = True
            return True
    st.error("❌ نام کاربری یا رمز عبور مدیر اشتباه است.")
    return False

def logout():
    # Keep theme but clear other session data
    current_theme = st.session_state.theme
    st.session_state.clear()
    st.session_state.initialized = True # Re-initialize after clearing
    st.session_state.theme = current_theme
    st.session_state.current_page = "login"
    st.rerun()

# --- User Management Functions (Admin Panel - Local) ---
def create_user_local(username, password, is_admin_user=False):
    users_data = load_users()
    if any(u["username"] == username for u in users_data["users"]) or \
       any(u["username"] == username for u in users_data["admin_users"]):
        st.warning("⚠️ کاربری با این نام کاربری از قبل وجود دارد.")
        return False
    
    if is_admin_user:
        users_data["admin_users"].append({"username": username, "password": password, "creation_time": datetime.now().isoformat()})
    else:
        users_data["users"].append({"username": username, "password": password, "creation_time": datetime.now().isoformat()})
    save_users(users_data)
    st.success(f"✅ کاربر '{username}' با موفقیت ایجاد شد.")
    return True

def delete_user_local(username, is_admin_user=False):
    users_data = load_users()
    target_list = "admin_users" if is_admin_user else "users"
    initial_len = len(users_data[target_list])
    users_data[target_list] = [u for u in users_data[target_list] if u["username"] != username]
    if len(users_data[target_list]) < initial_len:
        save_users(users_data)
        st.success(f"✅ کاربر '{username}' با موفقیت حذف شد.")
        return True
    st.warning("⚠️ کاربری با این نام کاربری یافت نشد.")
    return False

def list_users_local():
    users_data = load_users()
    all_users = []
    for user in users_data.get("users", []):
        all_users.append({"username": user["username"], "type": "عادی", "creation_time": user.get("creation_time", "N/A")})
    for admin in users_data.get("admin_users", []):
        all_users.append({"username": admin["username"], "type": "مدیر", "creation_time": admin.get("creation_time", "N/A")})
    return all_users

# --- Knowledge Base Loading Function (Optimized) ---
@st.cache_resource(ttl=3600) # Cache the loaded vector store for 1 hour
def load_knowledge_base_from_index(api_key, model_name="models/text-embedding-004"):
    """Loads the pre-built FAISS index from the local file system."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"🚨 پوشه ایندکس پایگاه دانش ('{FAISS_INDEX_PATH}') پیدا نشد. لطفاً ابتدا اسکریپت create_index.py را اجرا کرده و پوشه ایندکس را در کنار این فایل قرار دهید.")
        return None
    
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
        # The 'allow_dangerous_deserialization' flag is required for loading local FAISS indexes with newer LangChain versions
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"🚨 خطایی در بارگذاری پایگاه دانش از فایل ایندکس رخ داد: {e}")
        return None

# --- UI Pages ---

def login_page():
    st.header("ورود به دستیار دانش هوشمند سپاهان", divider="red")
    
    login_tab, admin_tab = st.tabs(["ورود کاربر", "ورود مدیر"])

    with login_tab:
        with st.form("user_login_form"):
            username = st.text_input("نام کاربری")
            password = st.text_input("رمز عبور", type="password")
            submitted = st.form_submit_button("ورود")
            if submitted:
                if user_login_local(username, password):
                    st.session_state.current_page = "chat"
                    st.rerun()

    with admin_tab:
        with st.form("admin_login_form"):
            admin_username = st.text_input("نام کاربری مدیر")
            admin_password = st.text_input("رمز عبور مدیر", type="password")
            admin_submitted = st.form_submit_button("ورود مدیر")
            if admin_submitted:
                if admin_login_local(admin_username, admin_password):
                    st.session_state.current_page = "admin"
                    st.rerun()

def admin_panel_page():
    st.header("🛠️ پنل مدیریت", divider="red")
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    st.warning("⚠️ **توجه:** تغییرات در لیست کاربران در فایل `users.json` ذخیره می‌شود و تا دیپلوی بعدی باقی می‌ماند. برای به‌روزرسانی پایگاه دانش، باید فایل `company_knowledge.pdf` را تغییر داده، اسکریپت `create_index.py` را مجدداً اجرا کرده و پوشه `faiss_index` جدید را آپلود کنید.")

    # --- User Management ---
    st.markdown("<h4>مدیریت کاربران</h4>", unsafe_allow_html=True)
    with st.expander("ایجاد کاربر جدید"):
        with st.form("create_user_form"):
            new_user_username = st.text_input("نام کاربری جدید")
            new_user_password = st.text_input("رمز عبور جدید", type="password")
            is_new_user_admin = st.checkbox("کاربر مدیر باشد؟")
            if st.form_submit_button("ایجاد کاربر"):
                if new_user_username and new_user_password:
                    create_user_local(new_user_username, new_user_password, is_new_user_admin)
                else:
                    st.warning("لطفاً نام کاربری و رمز عبور را وارد کنید.")

    with st.expander("حذف کاربر"):
        with st.form("delete_user_form"):
            users_list = list_users_local()
            usernames = [u['username'] for u in users_list if u['username'] != st.session_state.user_id] # Can't delete self
            delete_username = st.selectbox("انتخاب کاربر برای حذف", options=usernames)
            if st.form_submit_button("حذف کاربر"):
                if delete_username:
                    user_to_delete = next((u for u in users_list if u['username'] == delete_username), None)
                    if user_to_delete:
                        is_admin = user_to_delete['type'] == 'مدیر'
                        delete_user_local(delete_username, is_admin)
                else:
                    st.warning("لطفاً کاربری را برای حذف انتخاب کنید.")

    st.markdown("<h5>لیست کاربران موجود:</h5>", unsafe_allow_html=True)
    st.dataframe(list_users_local())

    # --- Knowledge Base Management ---
    st.markdown("<h4 style='margin-top: 30px;'>مدیریت پایگاه دانش</h4>", unsafe_allow_html=True)
    st.info("📚 سند اصلی پایگاه دانش از فایل `company_knowledge.pdf` ساخته شده و در پوشه `faiss_index` ذخیره شده است.")
    
    pdf_file_path_static = "company_knowledge.pdf"
    if os.path.exists(pdf_file_path_static):
        st.markdown(f"- **فایل منبع:** `{os.path.basename(pdf_file_path_static)}`")
        st.markdown(f"- **حجم فایل:** `{os.path.getsize(pdf_file_path_static) / 1024:.2f} KB`")
    else:
        st.error("🚨 فایل منبع 'company_knowledge.pdf' یافت نشد.")

def user_chat_page():
    st.sidebar.title("تنظیمات")
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    st.header("🧠 دستیار دانش هوشمند شرکت سپاهان", divider="red")

    # Load knowledge base (and cache it)
    vector_store = load_knowledge_base_from_index(google_api_key)

    if vector_store is None:
        st.error("🚨 پایگاه دانش بارگذاری نشد. لطفاً با مدیر سیستم تماس بگیرید.")
        return

    # Initialize LLMs once
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    
    # --- Chat Interface ---
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # This handles the new complex content structure
            content_data = message["content"]
            if content_data["type"] == "text":
                st.markdown(content_data["content"])
            elif content_data["type"] == "image":
                st.image(content_data["content"], caption="تصویر آپلود شده", use_column_width=True)
                st.markdown(content_data["text_content"])

    # --- User Input Handling ---
    if prompt := st.chat_input("سوال خود را در مورد دستورالعمل‌ها بپرسید..."):
        # Add user message to chat history
        user_message_content = {"type": "text", "content": prompt}
        st.session_state.messages.append({"role": "user", "content": user_message_content})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🚀 دستیار هوش مصنوعی در حال پردازش سوال شماست..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    full_response = response["result"]
                    st.markdown(full_response)
                    # Add assistant response to chat history
                    assistant_message = {"role": "assistant", "content": {"type": "text", "content": full_response}}
                    st.session_state.messages.append(assistant_message)
                except Exception as e:
                    error_message = f"⚠️ متاسفانه مشکلی در پردازش درخواست شما پیش آمده است. (خطا: {e})"
                    st.error(error_message)
                    assistant_message = {"role": "assistant", "content": {"type": "text", "content": error_message}}
                    st.session_state.messages.append(assistant_message)

# --- Main App Router ---
if st.session_state.current_page == "login":
    login_page()
elif st.session_state.current_page == "chat":
    user_chat_page()
elif st.session_state.current_page == "admin":
    admin_panel_page()
else:
    login_page() # Default to login page if state is invalid

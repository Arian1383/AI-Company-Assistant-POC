import streamlit as st
import os
import json
import tempfile
import base64
from datetime import datetime

# LangChain and AI related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- File Paths for Local Storage ---
USERS_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index"

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="دستیار دانش سپاهان",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- FONT LOADER (FIX) ---
# Inject Google Fonts link directly into the HTML head
st.markdown("""
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap" rel="stylesheet">
    </head>
""", unsafe_allow_html=True)


# --- Initialize session state variables safely ---
def initialize_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_page = "login"
        st.session_state.authenticated = False
        st.session_state.is_admin = False
        st.session_state.user_id = None
        st.session_state.messages = [
            {"role": "assistant", "content": {"type": "text", "content": "سلام! من دستیار هوشمند گروه صنعتی سپاهان هستم. چطور می‌توانم به شما کمک کنم؟"}}
        ]
        st.session_state.theme = "light"

initialize_session_state()

# --- Custom CSS Loader ---
def load_css(theme):
    css_file = "style.css" if theme == "light" else "style_dark.css"
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ فایل CSS '{css_file}' پیدا نشد. از تم پیش‌فرض Streamlit استفاده می‌شود.")

load_css(st.session_state.theme)

# --- API Key Management ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("🔑 خطای کلید API: کلید Google Gemini پیدا نشد. لطفاً آن را در Streamlit Secrets تنظیم کنید.")
    st.stop()

# --- Authentication Logic (Local) ---
def load_users():
    if not os.path.exists(USERS_FILE):
        default_users = {
            "users": [{"username": "Sepahan", "password": "Arian", "creation_time": datetime.now().isoformat()}],
            "admin_users": [{"username": "admin_sepahan", "password": "admin_pass", "creation_time": datetime.now().isoformat()}]
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=4)
        return default_users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users_data):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=4)

def validate_credentials(username, password, is_admin=False):
    users_data = load_users()
    user_type = "admin_users" if is_admin else "users"
    for user_info in users_data.get(user_type, []):
        if user_info["username"] == username and user_info["password"] == password:
            st.session_state.user_id = username
            st.session_state.authenticated = True
            st.session_state.is_admin = is_admin
            st.session_state.current_page = "admin" if is_admin else "chat"
            st.rerun()
    st.error("❌ نام کاربری یا رمز عبور اشتباه است.")

def logout():
    current_theme = st.session_state.theme
    # Reset all state variables to their initial values
    st.session_state.clear()
    initialize_session_state()
    st.session_state.theme = current_theme # Preserve theme choice
    st.rerun()

# --- User Management Logic (Admin) ---
def create_user(username, password, is_admin=False):
    if not username or not password:
        st.warning("لطفاً نام کاربری و رمز عبور را وارد کنید.")
        return
    users_data = load_users()
    if any(u["username"] == username for u in users_data["users"]) or \
       any(u["username"] == username for u in users_data["admin_users"]):
        st.warning("⚠️ کاربری با این نام کاربری از قبل وجود دارد.")
        return
    
    user_list = "admin_users" if is_admin else "users"
    users_data[user_list].append({"username": username, "password": password, "creation_time": datetime.now().isoformat()})
    save_users(users_data)
    st.success(f"✅ کاربر '{username}' با موفقیت ایجاد شد.")

def delete_user(username):
    users_data = load_users()
    # Find which list the user is in
    user_found = False
    for user_type in ["users", "admin_users"]:
        initial_len = len(users_data[user_type])
        users_data[user_type] = [u for u in users_data[user_type] if u["username"] != username]
        if len(users_data[user_type]) < initial_len:
            user_found = True
            break
    
    if user_found:
        save_users(users_data)
        st.success(f"✅ کاربر '{username}' با موفقیت حذف شد.")
        st.rerun()
    else:
        st.warning("⚠️ کاربری با این نام کاربری یافت نشد.")

# --- Knowledge Base Logic ---
@st.cache_resource(ttl=3600)
def load_knowledge_base_from_index(api_key):
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"🚨 پوشه ایندکس ('{FAISS_INDEX_PATH}') پیدا نشد. لطفاً با مدیر سیستم تماس بگیرید.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"🚨 خطایی در بارگذاری پایگاه دانش رخ داد: {e}")
        return None

# --- UI Rendering ---
def render_login_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    # You can add a logo here if you have one
    # st.image("path/to/your/logo.png", width=150) 
    st.markdown('<h2 class="login-title">ورود به دستیار هوشمند سپاهان</h2>', unsafe_allow_html=True)
    
    login_tab, admin_tab = st.tabs(["ورود کاربر", "ورود مدیر"])
    with login_tab:
        with st.form("user_login_form"):
            username = st.text_input("نام کاربری", placeholder="نام کاربری خود را وارد کنید")
            password = st.text_input("رمز عبور", type="password", placeholder="رمز عبور خود را وارد کنید")
            if st.form_submit_button("ورود"):
                validate_credentials(username, password, is_admin=False)
    with admin_tab:
        with st.form("admin_login_form"):
            admin_username = st.text_input("نام کاربری مدیر", placeholder="نام کاربری ادمین")
            admin_password = st.text_input("رمز عبور مدیر", type="password", placeholder="رمز عبور ادمین")
            if st.form_submit_button("ورود مدیر"):
                validate_credentials(admin_username, admin_password, is_admin=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_admin_panel():
    st.sidebar.title(f"خوش آمدید، {st.session_state.user_id}!")
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)
    
    st.markdown("<h1>🛠️ پنل مدیریت</h1>", unsafe_allow_html=True)
    
    management_tabs = st.tabs(["👤 مدیریت کاربران", "📚 مدیریت پایگاه دانش"])
    
    with management_tabs[0]:
        st.subheader("ایجاد کاربر جدید")
        with st.form("create_user_form"):
            new_user = st.text_input("نام کاربری جدید")
            new_pass = st.text_input("رمز عبور جدید", type="password")
            is_admin = st.checkbox("این کاربر مدیر باشد؟")
            if st.form_submit_button("ایجاد کاربر"):
                create_user(new_user, new_pass, is_admin)

        st.subheader("لیست کاربران")
        users = load_users()
        all_users = users.get("users", []) + users.get("admin_users", [])
        if not all_users:
            st.info("هیچ کاربری یافت نشد.")
        else:
            for user in all_users:
                cols = st.columns([0.6, 0.2, 0.2])
                cols[0].write(f"**نام کاربری:** {user['username']}")
                user_type = "مدیر" if user in users.get("admin_users", []) else "عادی"
                cols[1].write(f"**نوع:** {user_type}")
                if user['username'] != st.session_state.user_id: # Prevent self-deletion
                    if cols[2].button("حذف", key=f"del_{user['username']}", use_container_width=True):
                        delete_user(user['username'])

    with management_tabs[1]:
        st.subheader("وضعیت پایگاه دانش")
        st.info("📚 پایگاه دانش از فایل `company_knowledge.pdf` ساخته شده و در پوشه `faiss_index` ذخیره شده است.")
        st.warning("برای به‌روزرسانی، باید فایل PDF را جایگزین کرده، اسکریپت `create_index.py` را اجرا و پوشه `faiss_index` جدید را آپلود کنید.")
        
        pdf_path = "company_knowledge.pdf"
        if os.path.exists(pdf_path):
            st.markdown(f"- **فایل منبع:** `{os.path.basename(pdf_path)}`")
            st.markdown(f"- **حجم فایل:** `{os.path.getsize(pdf_path) / 1024:.2f} KB`")
            st.success("پایگاه دانش با موفقیت بارگذاری شده است.")
        else:
            st.error("🚨 فایل منبع 'company_knowledge.pdf' یافت نشد.")

def render_chat_page():
    st.sidebar.title(f"کاربر: {st.session_state.user_id}")
    if st.sidebar.toggle("فعال‌سازی تم تیره 🌙", key="theme_toggle", value=(st.session_state.theme == "dark")):
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    # Main chat container
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"]["content"])

    # Prompt suggestions
    st.markdown('<div class="prompt-suggestions">', unsafe_allow_html=True)
    cols = st.columns(3)
    suggestions = ["نحوه درخواست مرخصی", "ساعت کاری شرکت", "پشتیبانی IT"]
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, use_container_width=True):
            st.session_state.prompt = suggestion
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Handle chat input
    prompt = st.chat_input("سوال خود را بپرسید...", key="chat_input")
    if "prompt" in st.session_state and st.session_state.prompt:
        prompt = st.session_state.prompt
        st.session_state.prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": {"type": "text", "content": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🚀 در حال پردازش..."):
                vector_store = load_knowledge_base_from_index(google_api_key)
                if vector_store:
                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response.get("result", "متاسفانه پاسخی یافت نشد.")
                    except Exception as e:
                        full_response = f"⚠️ متاسفانه مشکلی در پردازش درخواست شما پیش آمده است. (خطا: {e})"
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "text", "content": full_response}})
                else:
                    st.error("خطا در اتصال به پایگاه دانش.")
                    st.session_state.messages.append({"role": "assistant", "content": {"type": "text", "content": "خطا در اتصال به پایگاه دانش."}})

# --- Main App Router ---
if st.session_state.authenticated:
    if st.session_state.is_admin:
        render_admin_panel()
    else:
        render_chat_page()
else:
    render_login_page()

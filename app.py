import streamlit as st
import os
import json
from datetime import datetime
import time

# LangChain and AI related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Constants ---
USERS_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index"
KNOWLEDGE_BASE_PDF = "company_knowledge.pdf"
CSS_FILE = "style.css"

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="دستیار دانش سپاهان",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Session State Initialization ---
def initialize_session_state():
    defaults = {
        "initialized": True, "current_page": "login", "authenticated": False,
        "is_admin": False, "user_id": None, "theme": "light",
        "messages": [{"role": "assistant", "content": "سلام! من دستیار هوشمند گروه صنعتی سپاهان هستم. چطور می‌توانم به شما کمک کنم؟"}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- CSS and Theme Management ---
def load_and_inject_css():
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # This is a trick to apply theme class to the body.
    st.markdown(f"""
        <script>
            document.body.classList.remove('light-theme', 'dark-theme');
            document.body.classList.add('{st.session_state.theme}-theme');
        </script>
    """, unsafe_allow_html=True)

load_and_inject_css()

# --- API Key Management ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("🔑 خطای کلید API: کلید Google Gemini پیدا نشد. لطفاً آن را در Streamlit Secrets تنظیم کنید.")
    st.stop()

# --- CORE LOGIC ---

def load_users():
    if not os.path.exists(USERS_FILE):
        default_users = {
            "users": [{"username": "Sepahan", "password": "Arian"}],
            "admin_users": [{"username": "admin_sepahan", "password": "admin_pass"}]
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=4)
        return default_users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

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
    theme = st.session_state.theme
    st.session_state.clear()
    initialize_session_state()
    st.session_state.theme = theme
    st.rerun()

@st.cache_resource(ttl=3600)
def load_knowledge_base_from_index(_api_key):
    if not os.path.exists(FAISS_INDEX_PATH): return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=_api_key)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"🚨 خطایی در بارگذاری پایگاه دانش رخ داد: {e}")
        return None

def rebuild_knowledge_base(pdf_file_bytes):
    """Saves the uploaded PDF and rebuilds the FAISS index."""
    with open(KNOWLEDGE_BASE_PDF, "wb") as f:
        f.write(pdf_file_bytes)
    
    loader = PyPDFLoader(KNOWLEDGE_BASE_PDF)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    
    # Clear the cache to force reload of the new index
    st.cache_resource.clear()

# --- UI RENDERING FUNCTIONS ---

def render_login_page():
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="login-title">ورود به دستیار هوشمند</h2>', unsafe_allow_html=True)
        
        login_tab, admin_tab = st.tabs(["ورود کاربر", "ورود مدیر"])
        with login_tab:
            with st.form("user_login_form"):
                username = st.text_input("نام کاربری", placeholder="نام کاربری خود را وارد کنید")
                password = st.text_input("رمز عبور", type="password", placeholder="رمز عبور خود را وارد کنید")
                if st.form_submit_button("ورود", use_container_width=True):
                    validate_credentials(username, password, is_admin=False)
        with admin_tab:
            with st.form("admin_login_form"):
                admin_username = st.text_input("نام کاربری مدیر", placeholder="نام کاربری ادمین")
                admin_password = st.text_input("رمز عبور مدیر", type="password", placeholder="رمز عبور ادمین")
                if st.form_submit_button("ورود مدیر", use_container_width=True):
                    validate_credentials(admin_username, admin_password, is_admin=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_admin_page():
    st.sidebar.title(f"پنل مدیریت")
    st.sidebar.caption(f"کاربر: {st.session_state.user_id}")
    st.sidebar.button("خروج", on_click=logout, use_container_width=True)

    st.title("🛠️ مدیریت سیستم")
    st.markdown("---")
    
    st.subheader("📚 مدیریت پایگاه دانش")
    st.info("در این بخش می‌توانید فایل PDF اصلی پایگاه دانش را جایگزین و سیستم را به‌روزرسانی کنید.")
    
    uploaded_file = st.file_uploader(
        "فایل PDF جدید را بارگذاری کنید",
        type="pdf",
        help="فایل جدید جایگزین فایل قبلی خواهد شد."
    )
    
    if uploaded_file is not None:
        if st.button("🚀 به‌روزرسانی پایگاه دانش", use_container_width=True):
            with st.spinner("لطفاً صبر کنید... در حال پردازش فایل و بازسازی پایگاه دانش. این فرآیند ممکن است چند دقیقه طول بکشد."):
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    rebuild_knowledge_base(pdf_bytes)
                    time.sleep(2) # Give a moment for user to see the message
                    st.success("✅ پایگاه دانش با موفقیت به‌روزرسانی شد!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ خطایی در هنگام به‌روزرسانی رخ داد: {e}")

def render_chat_page():
    with st.sidebar:
        st.title(f"کاربر: {st.session_state.user_id}")
        is_dark = st.session_state.theme == "dark"
        if st.toggle("فعال‌سازی تم تیره 🌙", value=is_dark):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        st.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    st.title("🧠 دستیار دانش سپاهان")
    
    # Use a container with a fixed height to make it scrollable
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("سوال خود را بپرسید..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("🚀 در حال پردازش..."):
                vector_store = load_knowledge_base_from_index(google_api_key)
                if vector_store:
                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response.get("result", "متاسفانه پاسخی یافت نشد.")
                    except Exception:
                        full_response = "⚠️ متاسفانه مشکلی در پردازش درخواست شما پیش آمده است."
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun() # Rerun to display the new messages
                else:
                    st.error("خطا در اتصال به پایگاه دانش.")

# --- Main App Router ---
if st.session_state.get("authenticated"):
    if st.session_state.get("is_admin"):
        render_admin_page()
    else:
        render_chat_page()
else:
    render_login_page()

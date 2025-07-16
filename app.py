import streamlit as st
import os
import json
from datetime import datetime

# LangChain and AI related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Constants ---
USERS_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index"
CSS_FILE = "style.css"

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="دستیار دانش سپاهان",
    page_icon="⚙️",  # More stable icon
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "initialized": True,
        "current_page": "login",
        "authenticated": False,
        "is_admin": False,
        "user_id": None,
        "theme": "light",
        "messages": [
            {"role": "assistant", "content": "سلام! من دستیار هوشمند گروه صنعتی سپاهان هستم. چطور می‌توانم به شما کمک کنم؟"}
        ]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- CSS and Theme Management ---
def load_and_inject_css():
    """Reads the CSS file and injects it into the app."""
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # This is a trick to apply theme class to the body. It might not be perfect but works.
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

# --- CORE LOGIC (Authentication, User Management, Knowledge Base) ---

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
    st.session_state.clear()
    initialize_session_state()
    st.session_state.theme = current_theme
    st.rerun()

@st.cache_resource(ttl=3600)
def load_knowledge_base_from_index(_api_key):
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=_api_key)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"🚨 خطایی در بارگذاری پایگاه دانش رخ داد: {e}")
        return None

# --- UI RENDERING FUNCTIONS ---

def render_login_page():
    # Use columns for robust centering
    _, center_col, _ = st.columns([1, 1.5, 1])
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

def render_chat_page():
    # --- Sidebar ---
    with st.sidebar:
        st.title(f"کاربر: {st.session_state.user_id}")
        
        # Theme toggle
        current_theme_is_dark = st.session_state.theme == "dark"
        if st.toggle("فعال‌سازی تم تیره 🌙", value=current_theme_is_dark):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"

        st.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    # --- Main Chat Area ---
    st.title("🧠 دستیار دانش سپاهان")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # --- Prompt Input ---
    if prompt := st.chat_input("سوال خود را بپرسید..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Immediately show user message
        with chat_container:
             with st.chat_message("user"):
                st.markdown(prompt)

        # Process and show assistant response
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
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("خطا در اتصال به پایگاه دانش.")
                    st.session_state.messages.append({"role": "assistant", "content": "خطا در اتصال به پایگاه دانش."})
        st.rerun()

# --- Main App Router ---
if st.session_state.get("authenticated"):
    render_chat_page() # Simplified: admin check can be added later if needed
else:
    render_login_page()

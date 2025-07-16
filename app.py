import streamlit as st
import os
import json
import time
from datetime import datetime

# LangChain and AI related imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Import Document here for load_knowledge_base

# --- Constants ---
USERS_FILE = "users.json"
FAISS_INDEX_PATH = "faiss_index"
KNOWLEDGE_BASE_PDF = "company_knowledge.pdf"
CSS_FILE_LIGHT = "style.css" # نام فایل CSS برای تم روشن
CSS_FILE_DARK = "style_dark.css" # نام فایل CSS برای تم تیره

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="دستیار دانش شرکت سپاهان",
    page_icon="🤖",
    layout="wide", # استفاده از layout="wide" برای فضای بیشتر
    initial_sidebar_state="collapsed"
)

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "initialized": True, "current_page": "login", "authenticated": False,
        "is_admin": False, "user_id": None, "theme": "light",
        "messages": [{"role": "assistant", "content": "سلام! من دستیار هوشمند گروه صنعتی سپاهان هستم. در این سامانه می‌توانید سوالات خود را در مورد دستورالعمل‌ها و رویه‌های شرکت بپرسید و پاسخ فوری دریافت کنید."}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- CSS and Theme Management ---
def set_theme(theme_name):
    st.session_state.theme = theme_name

def load_and_inject_css():
    """Reads the CSS file and injects it into the app."""
    css_file_path = CSS_FILE_LIGHT if st.session_state.theme == "light" else CSS_FILE_DARK
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ فایل CSS '{css_file_path}' پیدا نشد. لطفاً آن را در کنار 'app.py' قرار دهید.")
    
    # این اسکریپت CSS را به صورت مستقیم در تگ <body> تزریق می کند تا کنترل کامل روی تم داشته باشیم.
    # همچنین هدر پیش فرض Streamlit را به صورت قویتر پنهان می کند.
    st.markdown(f"""
        <style>
        /* پنهان کردن هدر پیش‌فرض Streamlit و فوتر "Made with Streamlit" */
        header {{ visibility: hidden; }}
        .stApp footer {{ visibility: hidden; }}
        /* اعمال کلاس تم به بدنه اصلی HTML برای کنترل کامل CSS */
        body {{
            background-color: var(--bg-color); /* از متغیر CSS استفاده می کند */
            color: var(--text-color); /* از متغیر CSS استفاده می کند */
            font-family: 'Vazirmatn', sans-serif;
            direction: rtl;
            text-align: right;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        body.light-theme {{
            --bg-color: #f8f9fa;
            --secondary-bg-color: #ffffff;
            --text-color: #212529;
            --header-color: #000000;
            --subtle-text-color: #65676b;
            --border-color: #ced0d4;
            --accent-color: #FFC107; /* Sepahan Yellow */
            --accent-text-color: #050505;
            --user-msg-bg: #eaf6ff; /* Light blue for user */
            --assistant-msg-bg: #ffffff; /* White for assistant */
            --shadow-color: rgba(0, 0, 0, 0.1);
            --input-bg: #f8f9fa;
            --input-border: #ced4da;
            --input-text-color: #343a40;
            --alert-bg-info: #e7f3ff;
            --alert-border-info: #FFC107;
            --alert-text-info: #212529;
        }}
        body.dark-theme {{
            --bg-color: #1a1a2e;
            --secondary-bg-color: #222831;
            --text-color: #e0e0e0;
            --header-color: #ffffff;
            --subtle-text-color: #b0b3b8;
            --border-color: #393e46;
            --accent-color: #e94560; /* Sepahan Red */
            --accent-text-color: #ffffff;
            --user-msg-bg: #0f3460; /* Dark blue for user */
            --assistant-msg-bg: #2c3e50; /* Dark gray for assistant */
            --shadow-color: rgba(0, 0, 0, 0.4);
            --input-bg: #1f2530;
            --input-border: #4a4a60;
            --input-text-color: #e0e0e0;
            --alert-bg-info: #0f3460;
            --alert-border-info: #FFC107;
            --alert-text-info: #f0f0f0;
        }}
        </style>
        <script>
            const body = window.parent.document.querySelector('body');
            body.classList.remove('light-theme', 'dark-theme');
            body.classList.add('{st.session_state.theme}-theme');
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
    theme = st.session_state.theme
    st.session_state.clear()
    initialize_session_state()
    st.session_state.theme = theme # حفظ تم انتخاب شده
    st.rerun()

def create_user(username, password, is_admin):
    if not username or not password:
        st.warning("لطفاً نام کاربری و رمز عبور را وارد کنید.")
        return
    users_data = load_users()
    if any(u["username"] == username for u in users_data["users"]) or \
       any(u["username"] == username for u in users_data["admin_users"]):
        st.warning("⚠️ کاربری با این نام کاربری از قبل وجود دارد.")
        return
    
    target_list = "admin_users" if is_admin else "users"
    users_data[target_list].append({"username": username, "password": password})
    save_users(users_data)
    st.success(f"✅ کاربر '{username}' با موفقیت ایجاد شد.")
    time.sleep(1)
    st.rerun()

def delete_user(username_to_delete):
    users_data = load_users()
    deleted = False
    for user_type in ["users", "admin_users"]:
        initial_len = len(users_data[user_type])
        users_data[user_type] = [u for u in users_data[user_type] if u['username'] != username_to_delete]
        if len(users_data[user_type]) < initial_len:
            save_users(users_data)
            deleted = True
            break
    if deleted:
        st.success(f"✅ کاربر '{username_to_delete}' با موفقیت حذف شد.")
        time.sleep(1)
        st.rerun()
    else:
        st.warning("⚠️ کاربری با این نام کاربری یافت نشد.")

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
    # Ensure FAISS_INDEX_PATH exists
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)

    with open(KNOWLEDGE_BASE_PDF, "wb") as f:
        f.write(pdf_file_bytes)
    
    loader = PyPDFLoader(KNOWLEDGE_BASE_PDF)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    
    st.cache_resource.clear() # Clear cache so load_knowledge_base_from_index reloads


# --- UI RENDERING FUNCTIONS ---

def render_login_page():
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="login-title">دستیار دانش گروه صنعتی سپاهان</h2>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">برای شروع، لطفاً با نام کاربری و رمز عبور خود وارد شوید. در صورت نداشتن حساب کاربری، با مدیر سیستم تماس بگیرید.</p>', unsafe_allow_html=True)
        
        # Theme selection buttons
        st.markdown("<div class='theme-selector'>", unsafe_allow_html=True)
        col_light, col_dark = st.columns(2)
        with col_light:
            if st.button("☀️ تم روشن", key="select_light_theme", use_container_width=True):
                set_theme("light")
                st.rerun()
        with col_dark:
            if st.button("🌙 تم تیره", key="select_dark_theme", use_container_width=True):
                set_theme("dark")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        login_tab, admin_tab = st.tabs(["ورود کاربر", "ورود مدیر"])
        with login_tab:
            with st.form("user_login_form"):
                username = st.text_input("نام کاربری", placeholder="نام کاربری خود را وارد کنید", label_visibility="collapsed")
                password = st.text_input("رمز عبور", type="password", placeholder="رمز عبور خود را وارد کنید", label_visibility="collapsed")
                if st.form_submit_button("ورود", use_container_width=True):
                    validate_credentials(username, password, is_admin=False)
        with admin_tab:
            with st.form("admin_login_form"):
                admin_username = st.text_input("نام کاربری مدیر", placeholder="نام کاربری ادمین", label_visibility="collapsed")
                admin_password = st.text_input("رمز عبور مدیر", type="password", placeholder="رمز عبور ادمین", label_visibility="collapsed")
                if st.form_submit_button("ورود مدیر", use_container_width=True):
                    validate_credentials(admin_username, admin_password, is_admin=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_admin_page():
    st.sidebar.title(f"پنل مدیریت")
    st.sidebar.caption(f"کاربر: {st.session_state.user_id}")
    is_dark = st.session_state.theme == "dark"
    if st.sidebar.toggle("فعال‌سازی تم تیره 🌙", value=is_dark):
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    st.sidebar.button("خروج", on_click=logout, use_container_width=True)

    st.title("🛠️ مدیریت سیستم")
    
    admin_tabs = st.tabs(["📚 مدیریت پایگاه دانش", "👤 مدیریت کاربران", "📊 لاگ‌های سیستم"])

    with admin_tabs[0]:
        st.subheader("به‌روزرسانی پایگاه دانش")
        st.info("در این بخش می‌توانید فایل PDF اصلی پایگاه دانش را جایگزین و سیستم را به‌روزرسانی کنید.")
        
        uploaded_file = st.file_uploader("فایل PDF جدید را بارگذاری کنید", type="pdf", label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("🚀 به‌روزرسانی و بازسازی پایگاه دانش", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="در حال آماده‌سازی...")
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    progress_bar.progress(25, text="فایل ذخیره شد. در حال پردازش و بازسازی پایگاه دانش...")
                    rebuild_knowledge_base(pdf_bytes)
                    progress_bar.progress(100, text="عملیات با موفقیت انجام شد!")
                    time.sleep(2)
                    st.success("✅ پایگاه دانش با موفقیت به‌روزرسانی شد!")
                    st.balloons()
                    progress_bar.empty()
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ خطایی در هنگام به‌روزرسانی رخ داد: {e}")

        st.markdown("<h5 class='admin-section-info'>اطلاعات سند اصلی:</h5>", unsafe_allow_html=True)
        if os.path.exists(KNOWLEDGE_BASE_PDF):
            st.markdown(f"- **نام فایل اصلی:** `{os.path.basename(KNOWLEDGE_BASE_PDF)}`")
            st.markdown(f"- **حجم فایل:** `{os.path.getsize(KNOWLEDGE_BASE_PDF) / (1024*1024):.2f} MB`")
            st.markdown("- **وضعیت:** بارگذاری شده در پایگاه دانش.")
        else:
            st.error("🚨 فایل 'company_knowledge.pdf' در مسیر برنامه یافت نشد.")


    with admin_tabs[1]:
        st.subheader("مدیریت کاربران")
        st.info("در این بخش می‌توانید کاربران عادی و مدیر را اضافه یا حذف کنید.")
        with st.form("create_user_form"):
            cols = st.columns([2, 2, 1])
            new_user = cols[0].text_input("نام کاربری جدید")
            new_pass = cols[1].text_input("رمز عبور جدید", type="password")
            is_admin = cols[2].checkbox("مدیر باشد؟")
            if st.form_submit_button("ایجاد کاربر", use_container_width=True):
                create_user(new_user, new_pass, is_admin)

        st.subheader("لیست کاربران موجود")
        users = load_users()
        all_users = users.get("users", []) + users.get("admin_users", [])
        if not all_users:
            st.info("هیچ کاربری یافت نشد.")
        else:
            for user in all_users:
                cols = st.columns([0.6, 0.2, 0.2])
                cols[0].write(f"**{user['username']}**")
                user_type = "مدیر" if user in users.get("admin_users", []) else "عادی"
                cols[1].markdown(f'<span class="user-role-badge role-{user_type.lower()}">{user_type}</span>', unsafe_allow_html=True)
                if user['username'] != st.session_state.user_id: # مدیر نتواند خودش را حذف کند
                    if cols[2].button("حذف", key=f"del_{user['username']}", use_container_width=True):
                        delete_user(user['username'])

    with admin_tabs[2]:
        st.subheader("لاگ‌های مکالمات کاربران")
        st.info("در این بخش می‌توانید تاریخچه مکالمات کاربران با دستیار هوشمند را مشاهده کنید.")
        st.warning("⚠️ **توجه:** لاگ‌ها در نسخه POC بدون دیتابیس آنلاین، دائمی نیستند و با هر بار دیپلوی یا ری‌استارت برنامه از بین می‌روند.")
        
        # In this local version, logs are not stored persistently.
        # We can display a dummy log or a message about non-persistence.
        st.markdown("""
        <div style="background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 10px; color: #f0f0f0;">
            <p style="font-size: 14px; margin-bottom: 5px;"><strong>لاگ‌ها در این نسخه ذخیره نمی‌شوند.</strong></p>
            <p style="font-size: 14px; margin-bottom: 0;">برای قابلیت لاگ دائمی، نیاز به اتصال به دیتابیس آنلاین (مانند Firebase Firestore) است.</p>
        </div>
        """, unsafe_allow_html=True)


def render_chat_page():
    with st.sidebar:
        st.title(f"کاربر: {st.session_state.user_id}")
        is_dark = st.session_state.theme == "dark"
        if st.toggle("فعال‌سازی تم تیره 🌙", value=is_dark):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        st.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    st.title("🧠 دستیار دانش هوشمند شرکت سپاهان")
    st.subheader("💡 سوالات خود را در مورد دستورالعمل‌ها و رویه‌های شرکت بپرسید.")

    st.info("💡 من اینجا هستم تا به سوالات شما بر اساس اسناد داخلی شرکت پاسخ دهم.")

    # Load knowledge base (and cache it)
    vector_store, _ = load_knowledge_base_local(google_api_key) # _ for all_chunks, not used here

    if vector_store is None:
        st.error("🚨 پایگاه دانش بارگذاری نشد. لطفاً با مدیر سیستم تماس بگیرید و اسناد را اضافه کنید.")
        return # Stop execution if KB not loaded

    retriever = vector_store.as_retriever()

    # --- User File/Image Upload for Context ---
    st.markdown("<h3 class='chat-section-header'>🖼️ افزودن فایل/تصویر به مکالمه</h3>", unsafe_allow_html=True)
    user_uploaded_context_file = st.file_uploader(
        "یک فایل PDF یا تصویر (JPG, PNG) برای افزودن به سوال فعلی آپلود کنید.",
        type=["pdf", "jpg", "jpeg", "png"],
        key="user_context_uploader",
    )

    # --- Connect to Google Gemini LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key) # Define multimodal_llm here

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="تصویر آپلود شده", use_column_width=True)
                st.markdown(message["text_content"]) # Display text description if available

    if prompt := st.chat_input("سوال خود را در مورد دستورالعمل‌ها بپرسید..."):
        user_message_content = {"type": "text", "content": prompt}
        gemini_prompt_parts = [{"text": prompt}]

        if user_uploaded_context_file:
            file_type = user_uploaded_context_file.type
            if "pdf" in file_type:
                st.info("در حال پردازش PDF آپلود شده برای سوال فعلی...")
                temp_pdf_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(user_uploaded_context_file.getvalue())
                        temp_pdf_path = temp_file.name
                    
                    loader = PyPDFLoader(temp_pdf_path)
                    pdf_docs = loader.load()
                    pdf_text = "\n".join([doc.page_content for doc in pdf_docs])
                    
                    user_message_content["content"] += f"\n\n(متن از فایل PDF: {pdf_text[:500]}...)" # Add snippet to user message
                    gemini_prompt_parts.append({"text": f"متن مرتبط از فایل PDF: {pdf_text}"})
                    st.success("PDF برای سوال فعلی پردازش شد.")
                except Exception as e:
                    st.error(f"خطا در پردازش PDF برای سوال فعلی: {e}")
                finally:
                    if temp_pdf_path and os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)

            elif "image" in file_type:
                st.info("در حال پردازش تصویر آپلود شده برای سوال فعلی...")
                base64_image = base64.b64encode(user_uploaded_context_file.getvalue()).decode('utf-8')
                user_message_content = {"type": "image", "content": f"data:{file_type};base64,{base64_image}", "text_content": prompt}
                
                gemini_prompt_parts = [
                    {"text": prompt},
                    {"inlineData": {"mimeType": file_type, "data": base64_image}}
                ]
                st.success("تصویر برای سوال فعلی پردازش شد.")
        
        st.session_state.messages.append({"role": "user", "content": user_message_content})
        with st.chat_message("user"):
            if user_message_content["type"] == "text":
                st.markdown(user_message_content["content"])
            elif user_message_content["type"] == "image":
                st.image(user_message_content["content"], caption="تصویر آپلود شده", use_column_width=True)
                st.markdown(user_message_content["text_content"])

        with st.chat_message("assistant"):
            with st.spinner("🚀 دستیار هوش مصنوعی در حال پردازش سوال شماست..."):
                try:
                    if user_uploaded_context_file and "image" in user_uploaded_context_file.type:
                        # For image input, direct API call is more flexible than RetrievalQA
                        # This bypasses RAG for image queries, as RAG is text-based.
                        # For true multimodal RAG, more complex setup is needed.
                        raw_response = multimodal_llm.invoke(gemini_prompt_parts) # Pass gemini_prompt_parts
                        full_response = raw_response.content
                    else:
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]

                    st.markdown(full_response)
                except Exception as e:
                    st.error(f"⚠️ متاسفانه در حال حاضر مشکلی در پردازش درخواست شما پیش آمده است. لطفاً لحظاتی دیگر دوباره تلاش کنید. (خطا: {e})")
            
            assistant_message_content = {"role": "assistant", "type": "text", "content": full_response}
            st.session_state.messages.append(assistant_message_content)

    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: #a0a0a0;'>نسخه آزمایشی v1.0 | تاریخ: {datetime.now().strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: #a0a0a0;'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)


# --- Main App Flow Control ---
if st.session_state.authenticated:
    if st.session_state.is_admin:
        admin_panel_page()
    else:
        user_chat_page()
else:
    # Login Page (common for both admin and user)
    st.title("🔐 ورود به دستیار دانش شرکت گروه صنعتی سپاهان")
    st.markdown("<hr style='border-top: 4px solid #FFC107; margin-bottom: 40px;'>", unsafe_allow_html=True) # Sepahan Yellow line
    st.info("👋 به سیستم دستیار هوشمند دانش شرکت خوش آمدید. لطفاً برای دسترسی، با نام کاربری و رمز عبور خود وارد شوید.")

    col1, col2, col3 = st.columns([1,2,1]) # For centering the form

    with col2: # Place form in the middle column
        login_type = st.radio("نوع ورود:", ("کاربر عادی", "مدیر سیستم"), horizontal=True)
        
        username = st.text_input("نام کاربری", key="login_username", help="نام کاربری پیش‌فرض: Sepahan (کاربر عادی) / admin_sepahan (مدیر)")
        password = st.text_input("رمز عبور", type="password", key="login_password", help="رمز عبور پیش‌فرض: Arian (کاربر عادی) / admin_pass (مدیر)")

        if st.button("ورود به سیستم 🚀"):
            if login_type == "کاربر عادی":
                if user_login_local(username, password):
                    st.success("✅ ورود موفقیت‌آمیز! در حال انتقال به دستیار هوشمند...")
                    st.rerun()
            elif login_type == "مدیر سیستم":
                if admin_login_local(username, password):
                    st.success("✅ ورود مدیر موفقیت‌آمیز! در حال انتقال به پنل مدیریت...")
                    st.rerun()

        st.caption("اگر دسترسی ندارید، لطفاً با بخش IT تماس بگیرید.")
    
    st.markdown("<hr style='border-top: 1px solid #e0e0e0; margin-top: 40px;'>", unsafe_allow_html=True) # Light gray line
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: #6c757d;'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)

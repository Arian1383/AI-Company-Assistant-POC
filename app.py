import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from datetime import datetime
import tempfile # برای ذخیره موقت فایل‌های آپلود شده
import base64 # برای مدیریت تصاویر
import json # برای ذخیره و خواندن کاربران و متادیتا فایل‌ها

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document # Import Document here for load_knowledge_base

# --- Global App ID (برای نامگذاری فایل‌های محلی) ---
# در این نسخه بدون Firebase، این فقط یک نام برای فایل‌های محلی است.
app_id = "sepahan-ai-assistant-local" 

# --- Initialize session state variables ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login' # صفحه پیش‌فرض ورود
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_email' not in st.session_state: # در این نسخه user_email همان username است
    st.session_state.user_email = None
if 'knowledge_vector_store' not in st.session_state:
    st.session_state.knowledge_vector_store = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'theme' not in st.session_state:
    st.session_state.theme = "light" # Default theme

# --- File Paths for Local Storage ---
USERS_FILE = "users.json"
# KNOWLEDGE_METADATA_FILE = "knowledge_metadata.json" # فعلا برای سادگی، مدیریت فایل از طریق GitHub است.


# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="دستیار دانش شرکت سپاهان",
    page_icon="🧠", # آیکون مغز برای حس AI
    layout="centered", # چیدمان مرکزی برای زیبایی بیشتر
    initial_sidebar_state="collapsed" # سایدبار به صورت پیش‌فرض بسته
)

# --- Theme Management ---
def set_theme(theme_name):
    st.session_state.theme = theme_name

# --- Custom CSS Loader ---
def load_css(theme):
    css_file = "style.css" if theme == "light" else "style_dark.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ فایل CSS '{css_file}' پیدا نشد. لطفاً آن را در کنار 'app.py' قرار دهید.")

# Apply selected theme CSS
load_css(st.session_state.theme)


# --- API Key و تنظیمات مدل ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError: # Changed from AttributeError to KeyError for secrets
    st.error("🔑 خطای کلید API: کلید Google Gemini پیدا نشد. لطفاً آن را در Streamlit Secrets تنظیم کنید.")
    st.stop()


# --- Authentication Functions (Local) ---
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"users": [], "admin_users": []}

def save_users(users_data):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=4)

def user_login_local(username, password):
    users_data = load_users()
    for user_info in users_data["users"]:
        if user_info["username"] == username and user_info["password"] == password:
            st.session_state.user_id = username # Using username as ID for simplicity
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
            st.session_state.user_id = username # Using username as ID for simplicity
            st.session_state.user_email = username
            st.session_state.authenticated = True
            st.session_state.is_admin = True
            return True
    st.error("❌ نام کاربری یا رمز عبور مدیر اشتباه است.")
    return False

def logout():
    st.session_state.clear()
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
    if is_admin_user:
        initial_len = len(users_data["admin_users"])
        users_data["admin_users"] = [u for u in users_data["admin_users"] if u["username"] != username]
        if len(users_data["admin_users"]) < initial_len:
            save_users(users_data)
            st.success(f"✅ کاربر مدیر '{username}' با موفقیت حذف شد.")
            return True
    else:
        initial_len = len(users_data["users"])
        users_data["users"] = [u for u in users_data["users"] if u["username"] != username]
        if len(users_data["users"]) < initial_len:
            save_users(users_data)
            st.success(f"✅ کاربر عادی '{username}' با موفقیت حذف شد.")
            return True
    st.warning("⚠️ کاربری با این نام کاربری یافت نشد.")
    return False

def list_users_local():
    users_data = load_users()
    all_users = []
    for user in users_data["users"]:
        all_users.append({"username": user["username"], "type": "عادی", "creation_time": user["creation_time"]})
    for admin in users_data["admin_users"]:
        all_users.append({"username": admin["username"], "type": "مدیر", "creation_time": admin["creation_time"]})
    return all_users


# --- PDF Processing Function (همان قبلی) ---
def process_pdf_for_rag(pdf_file, source_name):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if not documents:
            st.warning(f"⚠️ هشدار: هیچ متنی از فایل '{source_name}' استخراج نشد. لطفاً مطمئن شوید PDF قابل انتخاب متن است و خالی نیست.")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    except Exception as e:
        st.error(f"❌ خطایی در پردازش فایل '{source_name}' رخ داد: {e}.")
        return []

# --- Function to load and prepare knowledge base from local PDF ---
@st.cache_resource(ttl=3600) # Cache the vector store for 1 hour
def load_knowledge_base_local(api_key, model_name="models/text-embedding-004"):
    pdf_file_path_static = "company_knowledge.pdf"
    all_documents_for_rag = []

    if os.path.exists(pdf_file_path_static):
        try:
            loader = PyPDFLoader(pdf_file_path_static)
            all_documents_for_rag.extend(loader.load())
            st.success("✔️ فایل 'company_knowledge.pdf' با موفقیت بارگذاری شد.")
        except Exception as e:
            st.error(f"🚨 خطای بحرانی در خواندن فایل 'company_knowledge.pdf': {e}. لطفاً فایل را بررسی کنید.")
            return None, []
    else:
        st.error("🚨 فایل 'company_knowledge.pdf' پیدا نشد. لطفاً آن را در کنار 'app.py' قرار دهید.")
        return None, []

    if not all_documents_for_rag:
        st.error("🚨 هیچ متنی از فایل 'company_knowledge.pdf' استخراج نشد. لطفاً مطمئن شوید PDF قابل انتخاب متن است و خالی نیست.")
        return None, []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents_for_rag)

    embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    return vector_store, chunks


# --- Admin Panel Page (Local) ---
def admin_panel_page():
    st.header("🛠️ پنل مدیریت (نسخه POC)", divider="red")
    st.subheader("مدیریت اسناد و کاربران")

    # Theme Toggle
    st.sidebar.title("تنظیمات")
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        set_theme("dark" if st.session_state.theme == "light" else "light")
        st.rerun()

    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    st.warning("⚠️ **توجه:** این پنل مدیریت در نسخه POC بدون دیتابیس آنلاین است. تغییرات در اسناد و کاربران **دائمی نیستند** و با هر بار دیپلوی یا ری‌استارت برنامه از بین می‌روند.")
    
    # --- User Management (Local) ---
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>مدیریت کاربران محلی</h4>", unsafe_allow_html=True)
    
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

    st.markdown("<h5 style='text-align: right; color: #f0f0f0; margin-top: 20px; margin-bottom: 15px;'>لیست کاربران موجود:</h5>", unsafe_allow_html=True)
    users_list_data = list_users_local()
    if users_list_data:
        for user_info in users_list_data:
            st.write(f"- **نام کاربری:** {user_info['username']}, **نوع:** {user_info['type']}, **تاریخ ایجاد:** {user_info['creation_time']}")
    else:
        st.info("هیچ کاربری یافت نشد.")

    # --- Knowledge Base Management (Local) ---
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>مدیریت اسناد پایگاه دانش محلی</h4>", unsafe_allow_html=True)
    
    st.info("📚 سند اصلی پایگاه دانش، فایل 'company_knowledge.pdf' است که همراه برنامه دیپلوی شده است.")
    st.markdown("برای به‌روزرسانی سند اصلی، باید فایل 'company_knowledge.pdf' را در مخزن GitHub خود جایگزین کرده و برنامه را مجدداً دیپلوی کنید.")

    # نمایش اطلاعات فایل اصلی (اگر وجود داشته باشد)
    pdf_file_path_static = "company_knowledge.pdf"
    if os.path.exists(pdf_file_path_static):
        st.markdown(f"- **نام فایل اصلی:** `{os.path.basename(pdf_file_path_static)}`")
        st.markdown(f"- **حجم فایل:** `{os.path.getsize(pdf_file_path_static) / (1024*1024):.2f} MB`")
        st.markdown("- **وضعیت:** بارگذاری شده در پایگاه دانش.")
    else:
        st.error("🚨 فایل 'company_knowledge.pdf' در مسیر برنامه یافت نشد.")


# --- User Chat Page ---
def user_chat_page():
    st.sidebar.title("تنظیمات")
    # Theme Toggle
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        set_theme("dark" if st.session_state.theme == "light" else "light")
        st.rerun()
    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    st.header("🧠 دستیار دانش هوشمند شرکت سپاهان", divider="red")
    st.subheader("💡 سوالات خود را در مورد دستورالعمل‌ها و رویه‌های شرکت بپرسید.")

    st.info("💡 من اینجا هستم تا به سوالات شما بر اساس اسناد داخلی شرکت پاسخ دهم.")

    # Load knowledge base (and cache it)
    vector_store, all_chunks = load_knowledge_base_local(google_api_key) # استفاده از تابع جدید

    if vector_store is None:
        st.error("🚨 پایگاه دانش بارگذاری نشد. لطفاً با مدیر سیستم تماس بگیرید و اسناد را اضافه کنید.")
        return # Stop execution if KB not loaded

    retriever = vector_store.as_retriever()

    # --- User File/Image Upload for Context ---
    st.markdown("<h3 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 10px; margin-top: 40px; margin-bottom: 20px;'>🖼️ افزودن فایل/تصویر به مکالمه</h3>", unsafe_allow_html=True)
    user_uploaded_context_file = st.file_uploader(
        "یک فایل PDF یا تصویر (JPG, PNG) برای افزودن به سوال فعلی آپلود کنید.",
        type=["pdf", "jpg", "jpeg", "png"],
        key="user_context_uploader",
        help="این فایل فقط برای پاسخ به سوال فعلی استفاده می‌شود و به پایگاه دانش دائمی اضافه نمی‌شود."
    )

    # --- Connect to Google Gemini LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key) # تعریف multimodal_llm در اینجا

    # --- ساخت زنجیره سوال و جواب (RAG Chain) ---
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

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
                        multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key) # Added definition here
                        retrieved_docs = retriever.get_relevant_documents(prompt)
                        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
                        final_prompt_parts = [
                            {"text": f"سوال کاربر: {prompt}\n\nمتن مرتبط از پایگاه دانش: {context_text}"},
                            {"inlineData": {"mimeType": user_uploaded_context_file.type, "data": base64_image}}
                        ]
                        
                        raw_response = multimodal_llm.invoke(final_prompt_parts)
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
        password = st.text_input("رمز عبور", type="password", key="login_password", help="رمز عبور پیش‌فرض: Arian (کاربر عادی) / Arian (مدیر)")

        if st.button("ورود به سیستم 🚀"):
            if login_type == "کاربر عادی":
                # For regular users, attempt Firebase Auth login
                # Changed from firebase_auth.sign_in_with_email_and_password to a local check
                users_data = load_users()
                if user_login_local(username, password): # Use local login function
                    st.success("✅ ورود موفقیت‌آمیز! در حال انتقال به دستیار هوشمند...")
                    st.rerun()
                # Error message handled by user_login_local function
            elif login_type == "مدیر سیستم":
                if admin_login_local(username, password): # Use local admin login function
                    st.success("✅ ورود مدیر موفقیت‌آمیز! در حال انتقال به پنل مدیریت...")
                    st.rerun()
                # Error message handled by admin_login_local function

        st.caption("اگر دسترسی ندارید، لطفاً با بخش IT تماس بگیرید.")
    
    st.markdown("<hr style='border-top: 1px solid #e0e0e0; margin-top: 40px;'>", unsafe_allow_html=True) # Light gray line
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: #6c757d;'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)

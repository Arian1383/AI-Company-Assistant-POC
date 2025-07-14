
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from datetime import datetime
import tempfile
import json
import base64

# Firebase imports
from firebase_admin import credentials, initialize_app, auth, firestore

# --- Global Firebase App ID and Config (Provided by Canvas Environment) ---
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config_str = os.environ.get('__firebase_config', '{}')
initial_auth_token = os.environ.get('__initial_auth_token', '')

# --- Initialize session state variables ---
# اطمینان حاصل کنید که تمام متغیرهای session state در ابتدای اسکریپت مقداردهی اولیه شده‌اند.
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
if 'firebase_initialized' not in st.session_state:
    st.session_state.firebase_initialized = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'theme' not in st.session_state:
    st.session_state.theme = "light"


# --- Debugging Firebase Initialization (موقت برای عیب‌یابی) ---
# st.write("Debug: Firebase Initialized State:", st.session_state.firebase_initialized)
# st.write("Debug: db object:", db)
# st.write("Debug: firebase_auth object:", firebase_auth)


# Initialize Firebase Admin SDK (only once)
if not st.session_state.firebase_initialized:
    try:
        firebase_config = json.loads(firebase_config_str)
        if firebase_config:
            cred = credentials.Certificate(firebase_config)
            initialize_app(cred)
            st.session_state.firebase_initialized = True
            print("Firebase Admin SDK initialized successfully.")
        else:
            print("Firebase config is empty. Firebase Admin SDK not initialized.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")

# Firestore DB and Auth instances (re-get on rerun)
db = firestore.client() if st.session_state.firebase_initialized else None
firebase_auth = auth if st.session_state.firebase_initialized else None

# --- Firestore collection references (این بخش حیاتی است و باید اینجا باشد) ---
USERS_COLLECTION = db.collection(f"artifacts/{app_id}/users") if db else None
KNOWLEDGE_COLLECTION = db.collection(f"artifacts/{app_id}/public/data/knowledge_base_chunks") if db else None
LOGS_COLLECTION = db.collection(f"artifacts/{app_id}/public/data/chat_logs") if db else None


# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="دستیار دانش شرکت سپاهان",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
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


# --- Authentication Functions ---
def user_login_firebase(email, password):
    if firebase_auth:
        try:
            # Firebase Admin SDK does not directly sign in users with email/password.
            # This is a placeholder for a client-side login or custom token verification.
            # For POC, we assume if user exists, they can log in with any password.
            user = firebase_auth.get_user_by_email(email) # Check if user exists
            # In a real app, you'd verify password securely.
            
            st.session_state.user_id = user.uid
            st.session_state.user_email = email
            st.session_state.authenticated = True
            st.session_state.is_admin = False
            return True
        except firebase_auth.UserNotFoundError:
            st.error("❌ کاربری با این ایمیل یافت نشد.")
            return False
        except Exception as e:
            st.error(f"خطا در ورود کاربر: {e}")
            return False
    else:
        st.error("🚨 Firebase Auth فعال نیست. لطفاً با مدیر سیستم تماس بگیرید.")
        return False

def admin_login_hardcoded(username, password):
    ADMIN_USERNAME = "admin_sepahan"
    ADMIN_PASSWORD = "admin_pass" # CHANGE THIS IN PRODUCTION!
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.user_id = "admin_id_hardcoded"
        st.session_state.user_email = ADMIN_USERNAME
        st.session_state.authenticated = True
        st.session_state.is_admin = True
        return True
    else:
        st.error("❌ نام کاربری یا رمز عبور مدیر اشتباه است.")
        return False

def logout():
    st.session_state.clear()
    st.rerun()

# --- Firestore Data Management Functions ---
def add_knowledge_chunk_to_firestore(chunk_text, source_file, chunk_id=None):
    if KNOWLEDGE_COLLECTION:
        try:
            doc_ref = KNOWLEDGE_COLLECTION.document(chunk_id) if chunk_id else KNOWLEDGE_COLLECTION.document()
            doc_ref.set({
                "text": chunk_text,
                "source_file": source_file,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "uploader_id": st.session_state.user_id if 'user_id' in st.session_state else 'system',
                "uploader_email": st.session_state.user_email if 'user_email' in st.session_state else 'system'
            })
            return doc_ref.id
        except Exception as e:
            st.error(f"خطا در ذخیره سازی chunk در فایراستور: {e}")
            return None
    return None

def delete_knowledge_chunk_from_firestore(chunk_id):
    if KNOWLEDGE_COLLECTION:
        try:
            KNOWLEDGE_COLLECTION.document(chunk_id).delete()
            return True
        except Exception as e:
            st.error(f"خطا در حذف chunk از فایراستور: {e}")
            return False
    return False

def get_all_knowledge_chunks_from_firestore():
    if KNOWLEDGE_COLLECTION:
        try:
            docs = KNOWLEDGE_COLLECTION.stream()
            chunks_data = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                chunks_data.append(data)
            return chunks_data
        except Exception as e:
            st.error(f"خطا در دریافت chunks از فایراستور: {e}")
            return []
    return []

def log_chat_interaction(user_id, user_email, prompt, response):
    if LOGS_COLLECTION:
        try:
            LOGS_COLLECTION.add({
                "user_id": user_id,
                "user_email": user_email,
                "prompt": prompt,
                "response": response,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"Error logging chat interaction: {e}") # Print to console, not UI

# --- User Management Functions (Admin Panel) ---
def create_user(email, password):
    if firebase_auth:
        try:
            user = firebase_auth.create_user(email=email, password=password)
            st.success(f"کاربر '{email}' با موفقیت ایجاد شد. UID: {user.uid}")
            return True
        except Exception as e:
            st.error(f"خطا در ایجاد کاربر: {e}")
            return False
    else:
        st.warning("Firebase Auth فعال نیست.")
        return False

def delete_user(uid):
    if firebase_auth:
        try:
            firebase_auth.delete_user(uid)
            st.success(f"کاربر با UID '{uid}' با موفقیت حذف شد.")
            return True
        except Exception as e:
            st.error(f"خطا در حذف کاربر: {e}")
            return False
    else:
        st.warning("Firebase Auth فعال نیست.")
        return False

def list_users():
    if firebase_auth:
        try:
            users_page = firebase_auth.list_users()
            users_list = []
            for user in users_page.users:
                users_list.append({"uid": user.uid, "email": user.email, "creation_time": user.user_metadata.creation_timestamp})
            return users_list
        except Exception as e:
            st.error(f"خطا در لیست کردن کاربران: {e}")
            return []
    else:
        st.warning("Firebase Auth فعال نیست.")
        return []

# --- PDF Processing Function ---
def process_pdf_for_rag(pdf_file, source_name):
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.getvalue())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Ensure temporary file is deleted
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

# --- Function to load and prepare knowledge base from Firestore ---
@st.cache_resource(ttl=3600) # Cache the vector store for 1 hour
def load_knowledge_base(api_key, model_name="models/text-embedding-004"):
    all_chunks_data = get_all_knowledge_chunks_from_firestore()
    
    if not all_chunks_data:
        st.warning("⚠️ پایگاه دانش خالی است. لطفاً اسناد جدید اضافه کنید.")
        return None, []

    # Reconstruct Document objects from Firestore data
    documents_for_faiss = []
    for doc_data in all_chunks_data:
        # Ensure 'text' key exists before accessing
        if 'text' in doc_data:
            documents_for_faiss.append(st.runtime.state.SessionStateProxy._singleton.get_attribute("streamlit").Document( # Corrected Document import
                page_content=doc_data['text'],
                metadata={"source": doc_data.get('source_file', 'unknown'), "chunk_id": doc_data['id']}
            ))
        else:
            print(f"Warning: Document {doc_data.get('id', 'N/A')} in Firestore is missing 'text' field.")


    if not documents_for_faiss:
        st.error("🚨 خطای پردازش: هیچ chunk معتبری از پایگاه دانش استخراج نشد.")
        return None, []

    embeddings_model = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    vector_store = FAISS.from_documents(documents_for_faiss, embeddings_model)
    return vector_store, documents_for_faiss


# --- Admin Panel Page ---
def admin_panel_page():
    st.header("🛠️ پنل مدیریت", divider="red")
    st.subheader("مدیریت کاربران و اسناد")

    # Theme Toggle
    st.sidebar.title("تنظیمات")
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        set_theme("dark" if st.session_state.theme == "light" else "light")
        st.rerun()

    st.sidebar.button("خروج از سیستم 🚪", on_click=logout)

    # --- User Management ---
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>مدیریت کاربران</h4>", unsafe_allow_html=True)
    
    with st.expander("ایجاد کاربر جدید"):
        new_user_email = st.text_input("ایمیل کاربر جدید", key="new_user_email")
        new_user_password = st.text_input("رمز عبور کاربر جدید", type="password", key="new_user_password")
        if st.button("ایجاد کاربر"):
            if new_user_email and new_user_password:
                create_user(new_user_email, new_user_password)
            else:
                st.warning("لطفاً ایمیل و رمز عبور را وارد کنید.")
    
    with st.expander("حذف کاربر"):
        delete_user_uid = st.text_input("UID کاربر برای حذف", key="delete_user_uid")
        if st.button("حذف کاربر"):
            if delete_user_uid:
                delete_user(delete_user_uid)
            else:
                st.warning("لطفاً UID کاربر را وارد کنید.")

    st.markdown("<h5 style='text-align: right; color: #f0f0f0; margin-top: 20px; margin-bottom: 15px;'>لیست کاربران موجود:</h5>", unsafe_allow_html=True)
    users = list_users()
    if users:
        for user in users:
            st.write(f"- **UID:** {user['uid']}, **ایمیل:** {user['email']}, **تاریخ ایجاد:** {datetime.fromtimestamp(user['creation_time']/1000).strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("هیچ کاربری یافت نشد.")

    # --- Knowledge Base Management ---
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>مدیریت اسناد پایگاه دانش</h4>", unsafe_allow_html=True)
    
    uploaded_knowledge_files = st.file_uploader(
        "فایل‌های PDF جدید برای افزودن به پایگاه دانش (قابل انتخاب متن)",
        type="pdf",
        accept_multiple_files=True,
        key="admin_knowledge_uploader"
    )

    if uploaded_knowledge_files:
        for uploaded_file in uploaded_knowledge_files:
            with st.spinner(f"در حال پردازش '{uploaded_file.name}'..."):
                chunks = process_pdf_for_rag(uploaded_file, uploaded_file.name)
                if chunks:
                    for i, chunk in enumerate(chunks):
                        add_knowledge_chunk_to_firestore(chunk.page_content, uploaded_file.name, chunk_id=f"{uploaded_file.name}_{i}") # Use a unique ID
                    st.success(f"✔️ فایل '{uploaded_file.name}' و chunks آن با موفقیت به پایگاه دانش اضافه شد.")
                else:
                    st.error(f"❌ نتوانستیم متنی از فایل '{uploaded_file.name}' استخراج کنیم.")
        st.session_state.knowledge_vector_store = None # Invalidate cache to reload KB
        st.rerun() # Rerun to reflect changes

    st.markdown("<h5 style='text-align: right; color: #f0f0f0; margin-top: 20px; margin-bottom: 15px;'>اسناد موجود در پایگاه دانش:</h5>", unsafe_allow_html=True)
    current_knowledge_chunks = get_all_knowledge_chunks_from_firestore()
    if current_knowledge_chunks:
        for doc_data in current_knowledge_chunks:
            col_id, col_source, col_delete = st.columns([0.5, 2, 0.5])
            with col_source:
                st.write(f"- **فایل:** {doc_data['source_file']} (Chunk ID: {doc_data['id']})")
            with col_delete:
                if st.button("حذف", key=f"delete_chunk_{doc_data['id']}"):
                    if delete_knowledge_chunk_from_firestore(doc_data['id']):
                        st.success(f"Chunk '{doc_data['id']}' حذف شد.")
                        st.session_state.knowledge_vector_store = None # Invalidate cache
                        st.rerun()
    else:
        st.info("پایگاه دانش خالی است.")

    # --- View User Logs ---
    st.markdown("<h4 style='text-align: right; color: #f0f0f0; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 30px; margin-bottom: 20px;'>لاگ‌های مکالمات کاربران</h4>", unsafe_allow_html=True)
    
    if LOGS_COLLECTION:
        try:
            logs = LOGS_COLLECTION.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50).stream()
            log_entries = []
            for log in logs:
                log_data = log.to_dict()
                log_entries.append(log_data)
            
            if log_entries:
                for entry in log_entries:
                    st.markdown(f"""
                    <div style="background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 10px; color: #f0f0f0;">
                        <p style="font-size: 14px; margin-bottom: 5px;"><strong>کاربر:</strong> {entry.get('user_email', 'N/A')} ({entry.get('user_id', 'N/A')})</p>
                        <p style="font-size: 14px; margin-bottom: 5px;"><strong>زمان:</strong> {entry.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') if entry.get('timestamp') else 'N/A'}</p>
                        <p style="font-size: 15px; margin-bottom: 5px;"><strong>سوال:</strong> {entry.get('prompt', 'N/A')}</p>
                        <p style="font-size: 15px; margin-bottom: 0;"><strong>پاسخ:</strong> {entry.get('response', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("هیچ لاگی یافت نشد.")
        except Exception as e:
            st.error(f"خطا در دریافت لاگ‌ها: {e}")
    else:
        st.warning("Firebase Firestore فعال نیست.")


# --- User Chat Page ---
def user_chat_page():
    st.sidebar.title("تنظیمات")
    # Theme Toggle
    if st.sidebar.button("تغییر تم: روشن ☀️" if st.session_state.theme == "dark" else "تغییر تم: تیره 🌙"):
        set_theme("dark" if st.session_state.theme == "light" else "light")
        st.rerun()
    st.sidebar.button("خروج از سیستم 🚪", on_on_click=logout) # Changed on_click to on_on_click

    st.header("🧠 دستیار دانش هوشمند شرکت سپاهان", divider="red")
    st.subheader("💡 سوالات خود را در مورد دستورالعمل‌ها و رویه‌های شرکت بپرسید.")

    st.info("💡 من اینجا هستم تا به سوالات شما بر اساس اسناد داخلی شرکت پاسخ دهم.")

    # Load knowledge base (and cache it)
    vector_store, all_chunks = load_knowledge_base(google_api_key) # Pass API key

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
                # Process PDF for context
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
                # Process image for context (requires gemini-pro-vision or gemini-1.5-pro)
                st.info("در حال پردازش تصویر آپلود شده برای سوال فعلی...")
                base64_image = base64.b64encode(user_uploaded_context_file.getvalue()).decode('utf-8')
                user_message_content = {"type": "image", "content": f"data:{file_type};base64,{base64_image}", "text_content": prompt}
                
                # For image input, we need to use a multimodal model
                # Note: gemini-1.5-flash is multimodal, but older gemini-pro is not.
                # If using gemini-pro, this part needs adjustment or a different model for image input.
                # For now, assuming gemini-1.5-flash supports it.
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
                    # For image input, direct API call is more flexible than RetrievalQA
                    if user_uploaded_context_file and "image" in user_uploaded_context_file.type:
                        # Use a multimodal model directly for image input
                        # Note: This bypasses RAG for image queries, as RAG is text-based.
                        # For true multimodal RAG, more complex setup is needed.
                        multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                        
                        # Fetch relevant text from KB based on prompt first, then combine with image
                        retrieved_docs = retriever.get_relevant_documents(prompt)
                        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        
                        final_prompt_parts = [
                            {"text": f"سوال کاربر: {prompt}\n\nمتن مرتبط از پایگاه دانش: {context_text}"},
                            {"inlineData": {"mimeType": user_uploaded_context_file.type, "data": base64_image}}
                        ]
                        
                        raw_response = multimodal_llm.invoke(final_prompt_parts)
                        full_response = raw_response.content # Access content from AIMessage
                    else:
                        # Standard RAG for text-only queries (or PDF context)
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]

                    st.markdown(full_response)
                except Exception as e:
                    st.error(f"⚠️ متاسفانه در حال حاضر مشکلی در پردازش درخواست شما پیش آمده است. لطفاً لحظاتی دیگر دوباره تلاش کنید. (خطا: {e})")
            
            # Store assistant response in session state
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
                # For regular users, attempt Firebase Auth login
                if firebase_auth:
                    try:
                        # Authenticate user with email and password
                        user = firebase_auth.sign_in_with_email_and_password(username, password)
                        st.session_state.user_id = user['localId']
                        st.session_state.user_email = username
                        st.session_state.authenticated = True
                        st.session_state.is_admin = False
                        st.success("✅ ورود موفقیت‌آمیز! در حال انتقال به دستیار هوشمند...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ نام کاربری یا رمز عبور اشتباه است یا مشکلی در ارتباط با سرور وجود دارد: {e}")
                else:
                    st.error("🚨 Firebase فعال نیست. لطفاً با مدیر سیستم تماس بگیرید.")
            elif login_type == "مدیر سیستم":
                if admin_login_hardcoded(username, password):
                    st.success("✅ ورود مدیر موفقیت‌آمیز! در حال انتقال به پنل مدیریت...")
                    st.rerun()
                # Error message handled by admin_login_hardcoded function

        st.caption("اگر دسترسی ندارید، لطفاً با بخش IT تماس بگیرید.")
    
    st.markdown("<hr style='border-top: 1px solid #e0e0e0; margin-top: 40px;'>", unsafe_allow_html=True) # Light gray line
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: #6c757d;'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)

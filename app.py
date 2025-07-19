import streamlit as st
import os
import json
import time
from datetime import datetime
import base64
import tempfile
import docx # For Word files
import pandas as pd # For Excel files

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
KNOWLEDGE_BASE_PDF = "company_knowledge.pdf" # This will now be a placeholder, as we'll handle multiple types
CSS_FILE_LIGHT = "style_light.css" # نام فایل CSS برای تم روشن
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
        "initialized": True,
        "current_page": "login",
        "authenticated": False,
        "is_admin": False,
        "user_id": None,
        "theme": "light", # تم پیش‌فرض
        "messages": [{"role": "assistant", "type": "text", "content": "سلام! من دستیار هوشمند گروه صنعتی سپاهان هستم. در این سامانه می‌توانید سوالات خود را در مورد دستورالعمل‌ها و رویه‌های شرکت بپرسید و پاسخ فوری دریافت کنید."}],
        "page_history": [] # New: To store navigation history
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- CSS and Theme Management ---
def set_theme(theme_name):
    """Sets the theme in session state. Streamlit will rerun automatically."""
    st.session_state.theme = theme_name
    # Removed st.rerun() here as it's a no-op in on_change callbacks

def load_and_inject_css():
    """Reads the current theme's CSS file and injects it into the app."""
    css_file_path = CSS_FILE_LIGHT if st.session_state.theme == "light" else CSS_FILE_DARK
    
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ فایل CSS '{css_file_path}' پیدا نشد. لطفاً آن را در کنار 'app.py' قرار دهید.")
    
    # Inject general Streamlit overrides and font
    st.markdown(f"""
        <style>
        /* پنهان کردن هدر پیش‌فرض Streamlit و فوتر "Made with Streamlit" */
        header {{ visibility: hidden; }}
        .stApp footer {{ visibility: hidden; }}
        /* اعمال فونت و جهت متن به بدنه اصلی Streamlit */
        body {{
            font-family: 'Vazirmatn', sans-serif;
            direction: rtl;
            text-align: right;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        </style>
    """, unsafe_allow_html=True)

load_and_inject_css() # Call this after session state is initialized

# --- Global Theme Switcher (positioned using CSS) ---
# Render the toggle first, it will appear at the top of the Streamlit app content area.
# Then, CSS will move it to the desired fixed position and style it with icons.
is_dark_mode = st.session_state.theme == "dark"
# The on_change callback will update the theme and rerun the app
st.toggle("🌙", value=is_dark_mode, key="global_theme_toggle", help="تغییر تم (روشن/تیره)", label_visibility="hidden", on_change=lambda: set_theme("dark" if not is_dark_mode else "light"))

# Inject CSS for positioning the global theme switcher and styling its components
st.markdown(f"""
    <style>
    /* Target the st.toggle container */
    div[data-testid="stToggle"] {{
        position: fixed; /* Fixed position relative to viewport */
        top: 10px;
        left: 10px; /* Position to the left as per RTL layout */
        z-index: 9999; /* Ensure it's on top of other content */
        margin: 0 !important; /* Remove any default margins */
        padding: 0 !important; /* Remove any default paddings */
        background-color: var(--secondary-bg-color); /* Subtle background for the toggle box */
        border-radius: 15px; /* Rounded corners for the container */
        box-shadow: 0 2px 8px rgba(0,0,0,0.2); /* Small shadow */
        display: flex; /* To center the actual toggle inside */
        align-items: center;
        justify-content: center;
        width: 60px; /* Fixed width for the container */
        height: 30px; /* Fixed height for the container */
    }}

    /* Style the actual toggle switch (the track and thumb) */
    div[data-testid="stToggle"] .st-bo {{ /* This is the outer div of the toggle */
        width: 50px; /* Adjust width */
        height: 25px; /* Adjust height */
        border-radius: 15px; /* Make it rounded */
        background-color: var(--input-bg); /* Background of the toggle track */
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative; /* Needed for absolute positioning of icons */
    }}
    div[data-testid="stToggle"] .st-bo > div:first-child {{ /* This is the inner track/container for thumb */
        background-color: transparent; /* Track color is handled by .st-bo background */
        border-radius: 15px;
    }}
    div[data-testid="stToggle"] .st-bo > div:first-child > div:first-child {{ /* This is the thumb */
        width: 20px; /* Thumb size */
        height: 20px;
        background-color: var(--accent-color); /* Accent color for thumb */
        border-radius: 50%;
        margin: 2px; /* Small margin to keep it within track */
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        /* Add icon as background image */
        background-size: 80%; /* Adjust size of icon */
        background-repeat: no-repeat;
        background-position: center;
    }}

    /* Sun icon for light mode (toggle unchecked) */
    div[data-testid="stToggle"] input:not(:checked) + div > div > div {{
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23{f'{int(0.05 * 255):02x}'}{f'{int(0.05 * 255):02x}'}{f'{int(0.05 * 255):02x}'}'%3E%3Cpath d='M12 2.5a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2a.5.5 0 01.5-.5zM12 19.5a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2a.5.5 0 01.5-.5zM19.5 12a.5.5 0 01.5.5h2a.5.5 0 010 1h-2a.5.5 0 01-.5-.5v-1a.5.5 0 01.5-.5zM2.5 12a.5.5 0 01.5.5h2a.5.5 0 010 1h-2a.5.5 0 01-.5-.5v-1a.5.5 0 01.5-.5zM17.657 6.343a.5.5 0 01.354.146l1.414 1.414a.5.5 0 01-.707.707l-1.414-1.414a.5.5 0 01.353-.853zM4.929 17.657a.5.5 0 01.354.146l1.414 1.414a.5.5 0 01-.707.707l-1.414-1.414a.5.5 0 01.353-.853zM17.657 17.657a.5.5 0 01.354.146l1.414 1.414a.5.5 0 01-.707.707l-1.414-1.414a.5.5 0 01.353-.853zM4.929 6.343a.5.5 0 01.354.146l1.414 1.414a.5.5 0 01-.707.707l-1.414-1.414a.5.5 0 01.353-.853zM12 8a4 4 0 100 8 4 4 0 000-8z'/%3E%3C/svg%3E");
        background-color: var(--accent-color); /* Yellow for sun */
        transform: translateX(25px); /* Position to the right for light mode */
    }}

    /* Moon icon for dark mode (toggle checked) */
    div[data-testid="stToggle"] input:checked + div > div > div {{
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23{f'{int(0.9 * 255):02x}'}{f'{int(0.9 * 255):02x}'}{f'{int(0.9 * 255):02x}'}'%3E%3Cpath d='M12.3 4.5c.3-.3.7-.5 1.1-.5h.1c.4 0 .8.2 1.1.5.3.3.5.7.5 1.1v.1c0 .4-.2.8-.5 1.1-.3.3-.7.5-1.1.5h-.1c-.4 0-.8-.2-1.1-.5-.3-.3-.5-.7-.5-1.1v-.1c0-.4.2-.8.5-1.1zM12 2a10 10 0 100 20 10 10 0 000-20zM12 4a8 8 0 110 16 8 8 0 010-16zM13 5c-3.866 0-7 3.134-7 7s3.134 7 7 7c.302 0 .598-.02.89-.06a.5.5 0 01.61.61c-.3.292-.61.573-.93.837-1.17.96-2.61 1.52-4.17 1.52-4.97 0-9-4.03-9-9s4.03-9 9-9c1.56 0 3 .56 4.17 1.52.32.264.63.545.93.837a.5.5 0 01-.61.61c-.29-.04-.58-.06-.89-.06z'/%3E%3C/svg%3E");
        background-color: var(--accent-color); /* Teal for moon */
        transform: translateX(0); /* Position to the left for dark mode */
    }}

    /* Hide the default Streamlit label for the toggle */
    div[data-testid="stToggle"] label > div:last-child {{
        display: none;
    }}
    </style>
""", unsafe_allow_html=True)


# --- API Key Management ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("🔑 خطای کلید API: کلید Google Gemini پیدا نشد. لطفاً آن را در Streamlit Secrets تنظیم کنید.")
    st.stop()


# --- CORE LOGIC ---

def load_users():
    """Loads user data from JSON file. Creates default users if file doesn't exist."""
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
    """Saves user data to JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=4)

def navigate_to(page_name):
    """Navigates to a new page, pushing current page to history."""
    if st.session_state.current_page != page_name:
        st.session_state.page_history.append(st.session_state.current_page)
        st.session_state.current_page = page_name
        st.rerun() # Keep rerun here as it's a direct navigation action

def go_back():
    """Navigates back to the previous page in history."""
    if st.session_state.page_history:
        st.session_state.current_page = st.session_state.page_history.pop()
        st.rerun() # Keep rerun here as it's a direct navigation action
    else:
        st.warning("هیچ صفحه قبلی برای بازگشت وجود ندارد.")

# New function to navigate to the user's primary page (chat for user, admin for admin)
def go_to_main_page():
    """Navigates to the user's primary page (chat for regular user, admin for admin)."""
    if st.session_state.is_admin:
        navigate_to("admin")
    else:
        navigate_to("chat")

def validate_credentials(username, password, is_admin_attempt=False):
    """Validates user credentials and updates session state."""
    users_data = load_users()
    user_type_key = "admin_users" if is_admin_attempt else "users"
    
    for user_info in users_data.get(user_type_key, []):
        if user_info["username"] == username and user_info["password"] == password:
            st.session_state.user_id = username
            st.session_state.authenticated = True
            st.session_state.is_admin = is_admin_attempt
            
            # Use navigate_to for page change, which handles rerun
            target_page = "admin" if is_admin_attempt else "chat"
            navigate_to(target_page) 
            st.success("✅ ورود موفقیت‌آمیز! در حال انتقال...")
            time.sleep(1) # Keep sleep for user experience
            return True
    st.error("❌ نام کاربری یا رمز عبور اشتباه است.")
    return False

def logout():
    """Logs out the current user and resets session state."""
    theme_before_logout = st.session_state.theme # Preserve theme
    st.session_state.clear()
    initialize_session_state()
    st.session_state.theme = theme_before_logout # Restore theme
    st.rerun() # Keep rerun here as it's a direct navigation action

def create_user(username, password, is_admin):
    """Creates a new user (admin or regular)."""
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
    st.rerun() # Keep rerun here for immediate refresh of user list

def delete_user(username_to_delete):
    """Deletes a user by username."""
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
        st.rerun() # Keep rerun here for immediate refresh of user list
    else:
        st.warning("⚠️ کاربری با این نام کاربری یافت نشد.")

@st.cache_resource(ttl=3600)
def load_knowledge_base_from_index(_api_key):
    """Loads the FAISS knowledge base from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.warning("⚠️ پوشه پایگاه دانش FAISS یافت نشد. لطفاً ابتدا یک فایل PDF بارگذاری کنید.")
        return None, None # Return None for both vector_store and chunks
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=_api_key)
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # To get the chunks, we would typically need to re-process the PDF or save chunks separately.
        # For now, we'll just return the vector store. If chunks are needed for other purposes,
        # they should be handled during the rebuild_knowledge_base process.
        return vector_store, None 
    except Exception as e:
        st.error(f"🚨 خطایی در بارگذاری پایگاه دانش رخ داد: {e}")
        return None, None

def rebuild_knowledge_base(pdf_file_bytes):
    """Rebuilds the FAISS knowledge base from a new PDF file."""
    # Ensure FAISS_INDEX_PATH exists
    if not os.path.exists(FAISS_INDEX_PATH):
        os.makedirs(FAISS_INDEX_PATH)

    # Save the uploaded PDF to the designated path
    with open(KNOWLEDGE_BASE_PDF, "wb") as f:
        f.write(pdf_file_bytes)
    
    loader = PyPDFLoader(KNOWLEDGE_BASE_PDF)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Corrected typo here
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    
    st.cache_resource.clear() # Clear cache so load_knowledge_base_from_index reloads with new data


# --- UI RENDERING FUNCTIONS ---

def render_login_page():
    """Renders the login page."""
    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown('<h2 class="login-title">دستیار دانش گروه صنعتی سپاهان</h2>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">برای شروع، لطفاً با نام کاربری و رمز عبور خود وارد شوید. در صورت نداشتن حساب کاربری، با مدیر سیستم تماس بگیرید.</p>', unsafe_allow_html=True)
        
        login_tab, admin_tab = st.tabs(["ورود کاربر", "ورود مدیر"])
        with login_tab:
            with st.form("user_login_form"):
                username = st.text_input("نام کاربری", placeholder="نام کاربری خود را وارد کنید", label_visibility="collapsed")
                password = st.text_input("رمز عبور", type="password", placeholder="رمز عبور خود را وارد کنید", label_visibility="collapsed")
                if st.form_submit_button("ورود", use_container_width=True):
                    validate_credentials(username, password, is_admin_attempt=False)
        with admin_tab:
            with st.form("admin_login_form"):
                admin_username = st.text_input("نام کاربری مدیر", placeholder="نام کاربری ادمین", label_visibility="collapsed")
                admin_password = st.text_input("رمز عبور مدیر", type="password", placeholder="رمز عبور ادمین", label_visibility="collapsed")
                if st.form_submit_button("ورود مدیر", use_container_width=True):
                    validate_credentials(admin_username, admin_password, is_admin_attempt=True)
        st.markdown('</div>', unsafe_allow_html=True) # Closing login-card div


def render_admin_page():
    """Renders the admin panel page."""
    with st.sidebar:
        st.title(f"پنل مدیریت")
        st.caption(f"کاربر: {st.session_state.user_id}")
        
        st.markdown("---")
        if st.session_state.page_history:
            st.sidebar.button("🔙 بازگشت به صفحه قبلی", on_click=go_back, use_container_width=True)
        st.sidebar.button("🏠 بازگشت به صفحه اصلی", on_click=go_to_main_page, use_container_width=True) # New button
        
        if st.sidebar.button("⚙️ حساب کاربری من", key="my_account_btn_admin", use_container_width=True):
            navigate_to("user_account")
        st.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    st.title("🛠️ مدیریت سیستم")
    
    admin_tabs = st.tabs(["📚 مدیریت پایگاه دانش", "👤 مدیریت کاربران", "📊 لاگ‌های سیستم"])

    with admin_tabs[0]:
        st.subheader("به‌روزرسانی پایگاه دانش")
        st.info("در این بخش می‌توانید فایل PDF اصلی پایگاه دانش را جایگزین و سیستم را به‌روزرسانی کنید.")
        
        uploaded_file = st.file_uploader("فایل PDF جدید را بارگذاری کنید", type=["pdf", "docx", "xlsx"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("🚀 به‌روزرسانی و بازسازی پایگاه دانش", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text="در حال آماده‌سازی...")
                try:
                    file_bytes = uploaded_file.getvalue()
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    temp_file_path = None
                    if file_extension == ".pdf":
                        temp_file_path = "temp_knowledge.pdf"
                        with open(temp_file_path, "wb") as f:
                            f.write(file_bytes)
                        loader = PyPDFLoader(temp_file_path)
                    elif file_extension == ".docx":
                        temp_file_path = "temp_knowledge.docx"
                        with open(temp_file_path, "wb") as f:
                            f.write(file_bytes)
                        
                        # Read docx content
                        doc = docx.Document(temp_file_path)
                        full_text = []
                        for para in doc.paragraphs:
                            full_text.append(para.text)
                        # Create a single Document object for LangChain
                        documents = [Document(page_content="\n".join(full_text))]
                        loader = None # No specific LangChain loader needed if we extract text manually
                    elif file_extension == ".xlsx":
                        temp_file_path = "temp_knowledge.xlsx"
                        with open(temp_file_path, "wb") as f:
                            f.write(file_bytes)
                        
                        # Read xlsx content using pandas
                        xls = pd.ExcelFile(temp_file_path)
                        full_text = []
                        for sheet_name in xls.sheet_names:
                            df = xls.parse(sheet_name)
                            full_text.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")
                        documents = [Document(page_content="\n".join(full_text))]
                        loader = None # No specific LangChain loader needed if we extract text manually
                    else:
                        st.error("فرمت فایل پشتیبانی نمی‌شود.")
                        progress_bar.empty()
                        return

                    progress_bar.progress(25, text="فایل ذخیره شد. در حال پردازش و بازسازی پایگاه دانش...")
                    
                    if loader: # If it was a PDF and we used PyPDFLoader
                        documents = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=google_api_key)
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    vector_store.save_local(FAISS_INDEX_PATH)
                    
                    st.cache_resource.clear() # Clear cache so load_knowledge_base_from_index reloads
                    
                    progress_bar.progress(100, text="عملیات با موفقیت انجام شد!")
                    time.sleep(2)
                    st.success("✅ پایگاه دانش با موفقیت به‌روزرسانی شد!")
                    st.balloons()
                    progress_bar.empty()
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ خطایی در هنگام به‌روزرسانی رخ داد: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)


        st.markdown("<h5 class='admin-section-info'>اطلاعات سند اصلی:</h5>", unsafe_allow_html=True)
        # Display info about the last processed file, not just KNOWLEDGE_BASE_PDF
        if os.path.exists(KNOWLEDGE_BASE_PDF): # Assuming KNOWLEDGE_BASE_PDF is updated with the last file
            st.markdown(f"- **نام آخرین فایل پردازش شده:** `{os.path.basename(KNOWLEDGE_BASE_PDF)}`")
            st.markdown(f"- **حجم فایل:** `{os.path.getsize(KNOWLEDGE_BASE_PDF) / (1024*1024):.2f} MB`")
            st.markdown("- **وضعیت:** بارگذاری شده در پایگاه دانش.")
        else:
            st.info("💡 هنوز هیچ فایل پایگاه دانشی بارگذاری نشده است.")


    with admin_tabs[1]:
        st.subheader("مدیریت کاربران")
        st.info("در این بخش می‌توانید کاربران عادی و مدیر را اضافه یا حذف کنید.")
        with st.form("create_user_form"):
            cols = st.columns([2, 2, 1])
            new_user = cols[0].text_input("نام کاربری جدید", key="new_user_input")
            new_pass = cols[1].text_input("رمز عبور جدید", type="password", key="new_pass_input")
            is_admin_checkbox = cols[2].checkbox("مدیر باشد؟", key="is_admin_checkbox")
            if st.form_submit_button("ایجاد کاربر", use_container_width=True):
                create_user(new_user, new_pass, is_admin_checkbox)

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
                else:
                    cols[2].markdown("<span style='color: grey; font-size: 0.8em;'> (خودتان)</span>", unsafe_allow_html=True)


    with admin_tabs[2]:
        st.subheader("لاگ‌های مکالمات کاربران")
        st.info("در این بخش می‌توانید تاریخچه مکالمات کاربران با دستیار هوشمند را مشاهده کنید.")
        st.warning("⚠️ **توجه:** لاگ‌ها در نسخه POC بدون دیتابیس آنلاین، دائمی نیستند و با هر بار دیپلوی یا ری‌استارت برنامه از بین می‌روند.")
        
        st.markdown("""
        <div style="background-color: var(--secondary-bg-color); padding: 15px; border-radius: 10px; margin-bottom: 10px; color: var(--text-color); border: 1px solid var(--border-color);">
            <p style="font-size: 14px; margin-bottom: 5px;"><strong>لاگ‌ها در این نسخه ذخیره نمی‌شوند.</strong></p>
            <p style="font-size: 14px; margin-bottom: 0;'>برای قابلیت لاگ دائمی، نیاز به اتصال به دیتابیس آنلاین (مانند Firebase Firestore) است.</p>
        </div>
        """, unsafe_allow_html=True)


def render_chat_page():
    """Renders the main chat interface for regular users."""
    with st.sidebar:
        st.title(f"کاربر: {st.session_state.user_id}")
        
        st.markdown("---")
        if st.session_state.page_history:
            st.sidebar.button("🔙 بازگشت به صفحه قبلی", on_click=go_back, use_container_width=True)
        st.sidebar.button("🏠 بازگشت به صفحه اصلی", on_click=go_to_main_page, use_container_width=True) # New button

        if st.sidebar.button("⚙️ حساب کاربری من", key="my_account_btn_chat", use_container_width=True):
            navigate_to("user_account")
        st.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    st.title("🧠 دستیار دانش هوشمند شرکت سپاهان")
    st.subheader("💡 سوالات خود را در مورد دستورالعمل‌ها و رویه‌های شرکت بپرسید.")

    st.info("💡 من اینجا هستم تا به سوالات شما بر اساس اسناد داخلی شرکت پاسخ دهم.")

    # Load knowledge base (and cache it)
    vector_store, _ = load_knowledge_base_from_index(google_api_key) # _ for all_chunks, not used here

    if vector_store is None:
        st.error("🚨 پایگاه دانش بارگذاری نشد. لطفاً با مدیر سیستم تماس بگیرید و اسناد را اضافه کنید.")
        return # Stop execution if KB not loaded

    retriever = vector_store.as_retriever()
    
    # Initialize LLMs
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    multimodal_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    # Setup RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False # Set to True if you want to show sources
    )

    # --- User File/Image Upload for Context ---
    st.markdown("<h3 class='chat-section-header'>🖼️ افزودن فایل/تصویر به مکالمه</h3>", unsafe_allow_html=True)
    user_uploaded_context_file = st.file_uploader(
        "یک فایل PDF یا تصویر (JPG, PNG) برای افزودن به سوال فعلی آپلود کنید.",
        type=["pdf", "jpg", "jpeg", "png", "docx", "xlsx"], # Added docx, xlsx
        key="user_context_uploader",
    )

    # --- Chat Interface ---
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "image":
                st.image(message["content"], caption="تصویر آپلود شده", use_column_width=True)
                if "text_content" in message: # Display text description if available
                    st.markdown(message["text_content"])

    # Add a typing indicator placeholder
    if st.session_state.get("is_generating_response", False):
        with st.chat_message("assistant"):
            st.markdown('<div class="typing-indicator"><span>.</span><span>.</span><span>.</span></div>', unsafe_allow_html=True)


    # Accept user input
    if prompt := st.chat_input("سوال خود را در مورد دستورالعمل‌ها بپرسید..."):
        st.session_state.is_generating_response = True # Set flag when user inputs
        user_message_display = {"type": "text", "content": prompt}
        gemini_prompt_parts = [{"text": prompt}]

        # Handle uploaded file/image
        if user_uploaded_context_file:
            file_type = user_uploaded_context_file.type
            file_extension = os.path.splitext(user_uploaded_context_file.name)[1].lower()

            if file_extension == ".pdf":
                st.info("در حال پردازش PDF آپلود شده برای سوال فعلی...")
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(user_uploaded_context_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    loader = PyPDFLoader(temp_file_path)
                    pdf_docs = loader.load()
                    pdf_text = "\n".join([doc.page_content for doc in pdf_docs])
                    
                    user_message_display["content"] += f"\n\n(متن مرتبط از فایل PDF: {pdf_text[:500]}...)" # Add snippet to user message
                    gemini_prompt_parts.append({"text": f"متن مرتبط از فایل PDF: {pdf_text}"})
                    st.success("PDF برای سوال فعلی پردازش شد.")
                except Exception as e:
                    st.error(f"خطا در پردازش PDF برای سوال فعلی: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            elif file_extension == ".docx":
                st.info("در حال پردازش فایل Word آپلود شده برای سوال فعلی...")
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                        temp_file.write(user_uploaded_context_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    doc = docx.Document(temp_file_path)
                    full_text = []
                    for para in doc.paragraphs:
                        full_text.append(para.text)
                    doc_text = "\n".join(full_text)
                    
                    user_message_display["content"] += f"\n\n(متن مرتبط از فایل Word: {doc_text[:500]}...)"
                    gemini_prompt_parts.append({"text": f"متن مرتبط از فایل Word: {doc_text}"})
                    st.success("فایل Word برای سوال فعلی پردازش شد.")
                except Exception as e:
                    st.error(f"خطا در پردازش فایل Word برای سوال فعلی: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            elif file_extension == ".xlsx":
                st.info("در حال پردازش فایل Excel آپلود شده برای سوال فعلی...")
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                        temp_file.write(user_uploaded_context_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    xls = pd.ExcelFile(temp_file_path)
                    full_text = []
                    for sheet_name in xls.sheet_names:
                        df = xls.parse(sheet_name)
                        full_text.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")
                    excel_text = "\n".join(full_text)
                    
                    user_message_display["content"] += f"\n\n(متن مرتبط از فایل Excel: {excel_text[:500]}...)"
                    gemini_prompt_parts.append({"text": f"متن مرتبط از فایل Excel: {excel_text}"})
                    st.success("فایل Excel برای سوال فعلی پردازش شد.")
                except Exception as e:
                    st.error(f"خطا در پردازش فایل Excel برای سوال فعلی: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            elif "image" in file_type:
                st.info("در حال پردازش تصویر آپلود شده برای سوال فعلی...")
                base64_image = base64.b64encode(user_uploaded_context_file.getvalue()).decode('utf-8')
                user_message_display = {"type": "image", "content": f"data:{file_type};base64,{base64_image}", "text_content": prompt}
                
                gemini_prompt_parts = [
                    {"text": prompt},
                    {"inlineData": {"mimeType": file_type, "data": base64_image}}
                ]
                st.success("تصویر برای سوال فعلی پردازش شد.")
            else:
                st.warning("فرمت فایل آپلود شده پشتیبانی نمی‌شود.")
        
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", **user_message_display})
        with st.chat_message("user"):
            if user_message_display["type"] == "text":
                st.markdown(user_message_display["content"])
            elif user_message_display["type"] == "image":
                st.image(user_message_display["content"], caption="تصویر آپلود شده", use_column_width=True)
                st.markdown(user_message_display["text_content"])

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🚀 دستیار هوش مصنوعی در حال پردازش سوال شماست..."):
                try:
                    full_response = ""
                    if user_uploaded_context_file and ("image" in file_type): # Check file_type here
                        # For image input, use multimodal_llm directly
                        raw_response = multimodal_llm.invoke(gemini_prompt_parts)
                        full_response = raw_response.content
                    else:
                        # For text, PDF, DOCX, XLSX input, use qa_chain (RAG)
                        # Join text parts if any document was uploaded
                        combined_prompt = prompt
                        if len(gemini_prompt_parts) > 1: # If additional text parts were added (from PDF/DOCX/XLSX)
                            combined_prompt = "\n\n".join([p["text"] for p in gemini_prompt_parts if "text" in p])
                        
                        response = qa_chain.invoke({"query": combined_prompt})
                        full_response = response["result"]

                    st.markdown(full_response)
                except Exception as e:
                    st.error(f"⚠️ متاسفانه در حال حاضر مشکلی در پردازش درخواست شما پیش آمده است. لطفاً لحظاتی دیگر دوباره تلاش کنید. (خطا: {e})")
                finally:
                    st.session_state.is_generating_response = False # Reset flag after generation

            assistant_message_content = {"role": "assistant", "type": "text", "content": full_response}
            st.session_state.messages.append(assistant_message_content)
        
        # JavaScript to scroll to the bottom after new messages are added
        st.markdown("""
            <script>
                var chatContainer = parent.document.querySelector('.main .block-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            </script>
        """, unsafe_allow_html=True)


    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: var(--subtle-text-color);'>نسخه آزمایشی v1.0 | تاریخ: {datetime.now().strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: var(--subtle-text-color);'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)


def render_user_account_page():
    """Renders the user account management page (e.g., change password)."""
    with st.sidebar:
        st.title(f"حساب کاربری: {st.session_state.user_id}")
        st.markdown("---")
        if st.session_state.page_history:
            st.sidebar.button("🔙 بازگشت به صفحه قبلی", on_click=go_back, use_container_width=True)
        st.sidebar.button("🏠 بازگشت به صفحه اصلی", on_click=go_to_main_page, use_container_width=True) # New button
        st.button("خروج از سیستم 🚪", on_click=logout, use_container_width=True)

    st.title("👤 مدیریت حساب کاربری")
    st.info(f"شما به عنوان **{st.session_state.user_id}** وارد شده‌اید.")

    st.subheader("تغییر رمز عبور")
    with st.form("change_password_form"):
        current_password = st.text_input("رمز عبور فعلی", type="password", key="current_pass_input")
        new_password = st.text_input("رمز عبور جدید", type="password", key="new_pass_input")
        confirm_new_password = st.text_input("تکرار رمز عبور جدید", type="password", key="confirm_new_pass_input")

        if st.form_submit_button("تغییر رمز عبور", use_container_width=True, type="primary"):
            users_data = load_users()
            user_type_key = "admin_users" if st.session_state.is_admin else "users"
            
            user_found = False
            for user_info in users_data.get(user_type_key, []):
                if user_info["username"] == st.session_state.user_id:
                    user_found = True
                    if user_info["password"] == current_password:
                        if new_password == confirm_new_password:
                            if new_password and len(new_password) >= 4: # Basic validation
                                user_info["password"] = new_password
                                save_users(users_data)
                                st.success("✅ رمز عبور با موفقیت تغییر یافت.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("⚠️ رمز عبور جدید حداقل باید 4 کاراکتر باشد.")
                        else:
                            st.warning("⚠️ رمز عبور جدید و تکرار آن مطابقت ندارند.")
                    else:
                        st.error("❌ رمز عبور فعلی اشتباه است.")
                    break
            if not user_found:
                st.error("خطا: کاربر جاری در پایگاه داده یافت نشد. لطفاً دوباره وارد شوید.")

    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: var(--subtle-text-color);'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)


# --- Main App Flow Control ---
if st.session_state.authenticated:
    if st.session_state.current_page == "admin":
        render_admin_page()
    elif st.session_state.current_page == "chat":
        render_chat_page()
    elif st.session_state.current_page == "user_account":
        render_user_account_page()
    else:
        # Fallback for unexpected current_page values
        st.error("خطای ناوبری: صفحه نامعتبر.")
        logout() # Force logout to reset
else:
    render_login_page()
    st.markdown("<hr style='border-top: 1px solid var(--border-color); margin-top: 40px;'>", unsafe_allow_html=True) # Light gray line
    st.markdown(f"<p style='text-align: center; font-size: 13px; color: var(--subtle-text-color);'>&copy; {datetime.now().year} گروه صنعتی سپاهان. تمامی حقوق محفوظ است.</p>", unsafe_allow_html=True)

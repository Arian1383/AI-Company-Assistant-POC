# test_api.py
import google.generativeai as genai
import os

# --- مرحله ۱: کلید API جدید خود را اینجا قرار دهید ---
# این کلید باید همان کلیدی باشد که در create_index.py استفاده کردید.
GOOGLE_API_KEY = "AIzaSyBzAKJqZKdQQmkuVbPoA2AvLGqZ7uXjUhI"

# --- بررسی اولیه ---
if GOOGLE_API_KEY == "AIzaSyBzAKJqZKdQQmkuVbPoA2AvLGqZ7uXjUhI":
    print("🚨 خطا: لطفاً قبل از اجرا، کلید API جدید خود را در متغیر GOOGLE_API_KEY قرار دهید.")
    exit()

try:
    # --- مرحله ۲: پیکربندی API با کلید شما ---
    print("⏳ در حال تنظیم و تست کلید API...")
    genai.configure(api_key=GOOGLE_API_KEY)

    # --- مرحله ۳: درخواست لیست مدل‌ها از گوگل ---
    # این یک درخواست ساده برای تست کردن ارتباط و اعتبار کلید است.
    print("📡 در حال ارسال درخواست تست به سرورهای گوگل...")
    
    model_list = [m for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
    
    # اگر کد به اینجا برسد و خطا ندهد، یعنی کلید شما معتبر است.
    print("\n✅✅✅ تبریک! کلید API شما معتبر است و با موفقیت کار می‌کند.")
    print("لیست مدل‌های Embedding موجود:")
    for model in model_list:
        print(f"- {model.name}")

except Exception as e:
    print("\n❌❌❌ متاسفانه تست ناموفق بود.")
    print("خطای دریافت شده مستقیماً از سرور گوگل است:")
    print(f"\n--- متن خطا ---\n{e}\n-----------------")
    print("\n💡 این یعنی مشکل قطعاً از یکی از موارد زیر است:")
    print("1. کلید API به درستی کپی نشده است.")
    print("2. سرویس 'Generative Language API' برای پروژه شما در Google Cloud فعال نیست.")
    print("3. یک حساب پرداخت (Billing Account) به پروژه شما متصل نیست.")
    print("4. کلید شما محدودیت‌هایی (Restrictions) دارد که اجازه دسترسی را نمی‌دهد.")


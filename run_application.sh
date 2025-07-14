#!/bin/bash

# رفتن به پوشه ریشه پروژه (اگر از جای دیگری اجرا شود)
cd "$(dirname "$0")"

echo "🚀 در حال همگام‌سازی تغییرات با GitHub..."

# اضافه کردن تمام فایل‌ها برای Commit
git add .

# ایجاد یک Commit جدید (اگر تغییری وجود داشته باشد)
# از git diff-index برای بررسی وجود تغییرات استفاده می کنیم
if ! git diff-index --quiet HEAD --; then
    git commit -m "Automated commit: Syncing changes before running application"
    if [ $? -ne 0 ]; then # بررسی موفقیت آمیز بودن commit
        echo "❌ خطا در Commit کردن تغییرات. لطفا به صورت دستی بررسی کنید."
        exit 1 # خروج با خطا
    fi
else
    echo "✅ هیچ تغییری برای Commit کردن وجود ندارد."
fi

# ارسال تمام تغییرات به GitHub (با اجبار)
git push -u origin main --force
if [ $? -ne 0 ]; then # بررسی موفقیت آمیز بودن push
    echo "❌ خطا در Push کردن تغییرات به GitHub. لطفا به صورت دستی بررسی کنید."
    exit 1 # خروج با خطا
fi

echo "✅ تغییرات با موفقیت به GitHub منتقل شد."

# فعال کردن محیط مجازی
echo "⚙️ در حال فعال‌سازی محیط مجازی..."
source .venv/bin/activate
if [ $? -ne 0 ]; then # بررسی موفقیت آمیز بودن فعال سازی محیط مجازی
    echo "❌ خطا در فعال‌سازی محیط مجازی. مطمئن شوید پوشه .venv وجود دارد و pip install -r requirements.txt اجرا شده است."
    exit 1
fi

# اجرای برنامه Streamlit
echo "✨ در حال اجرای اپلیکیشن Streamlit..."
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
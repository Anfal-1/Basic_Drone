from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import cv2
import numpy as np
import os
import magic  # مكتبة لفحص نوع الملف الحقيقي
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

# ✅ تحميل الموديل المدرب مسبقًا
model = YOLO("yolov8n.pt")  # تأكد من أن لديك هذا الموديل

# ✅ إنشاء تطبيق FastAPI
app = FastAPI()

# ✅ تقييد الاستضافة لمنع الطلبات غير المصرح بها
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["theecotrack.com", "localhost"])

# ✅ تفعيل ضغط GZIP لتقليل حجم البيانات المرسلة
app.add_middleware(GZipMiddleware)

# ✅ تفعيل CORS للسماح بطلبات من الويب
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://theecotrack.com"],  # يمكن تعديلها حسب الحاجة
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ✅ إعداد Rate Limiting لمنع الطلبات الزائدة
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ✅ الصفحة الرئيسية لاختبار تشغيل API
@app.get("/")
def home():
    return {"message": "✅ YOLOv8 API is Running!"}

# ✅ السماح فقط بملفات الصور
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 5

def validate_file(file: UploadFile):
    # التحقق من الامتداد
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="❌ الملف غير مدعوم. يُسمح فقط بملفات JPG, JPEG, PNG.")

    # التحقق من حجم الملف
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell() / (1024 * 1024)  # التحويل إلى MB
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"❌ الحد الأقصى لحجم الملف {MAX_FILE_SIZE_MB}MB فقط.")

    # التحقق من نوع الملف الحقيقي
    mime = magic.Magic(mime=True)
    file_type = mime.from_buffer(file.file.read(2048))
    file.file.seek(0)
    if not file_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="❌ الملف ليس صورة صحيحة.")

    return file

# ✅ API لتحليل الصور بدون قاعدة بيانات
@app.post("/detect/")
@limiter.limit("5/minute")  # السماح بـ 5 طلبات فقط في الدقيقة لكل مستخدم
async def detect_objects(file: UploadFile = File(...)):
    try:
        file = validate_file(file)

        # ✅ قراءة الصورة وتحويلها إلى NumPy Array
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ✅ تشغيل YOLOv8 على الصورة
        results = model(image)

        # ✅ استخراج البيانات من النتائج
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # إحداثيات الصندوق
                conf = float(box.conf[0])  # نسبة الثقة
                label = result.names[int(box.cls[0])]  # اسم الكائن المكتشف

                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

        return {"status": "success", "detections": detections}

    except Exception as e:
        return {"error": f"❌ حدث خطأ أثناء التحليل: {str(e)}"}

# ✅ تشغيل السيرفر
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
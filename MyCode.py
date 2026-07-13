import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# 1. تحميل موديل TFLite وتخصيص الذاكرة له
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# الحصول على تفاصيل المدخلات والمخرجات للموديل
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. تحميل ملف التصنيفات
class_names = open("labels.txt", "r").readlines()

# 3. معالجة وتجهيز الصورة
image = Image.open("TestImage.jpg").convert("RGB")
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# تحويل الصورة إلى مصفوفة وتطبيعها (Normalization)
image_array = np.asarray(image, dtype=np.float32)
normalized_image_array = (image_array / 127.5) - 1.0

# إضافة البُعد الصفري ليصبح (1, 224, 224, 3)
input_data = np.expand_dims(normalized_image_array, axis=0)

# 4. تغذية الموديل بالصورة وتشغيل التنبؤ
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# الحصول على النتيجة
prediction = interpreter.get_tensor(output_details[0]['index'])
index = np.argmax(prediction[0])
class_name = class_names[index]
confidence_score = prediction[0][index]

# 5. طباعة النتيجة النهائية
print("\n" + "="*30)
print("Class:", class_name[2:].strip())
print(f"Confidence Score: {round(confidence_score * 100, 2)}%")
print("="*30 + "\n")
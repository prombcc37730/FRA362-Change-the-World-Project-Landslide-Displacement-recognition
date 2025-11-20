
import cv2
import os

# --------------------------
# 1) ใส่ path รูปต้นฉบับ
# --------------------------
input_path = r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\Equalization_image\083142_0831_2025.jpg"

# --------------------------
# 2) ใส่ path + ชื่อไฟล์ที่จะเซฟ
# --------------------------
output_path = r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\edge_detection_image\canny_083142_0831_2025.jpg"

# โหลดภาพแบบ grayscale
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# เช็คว่าภาพโหลดได้ไหม
if img is None:
    print("❌ ไม่พบไฟล์ภาพ! ตรวจสอบ path อีกครั้ง")
    exit()

# --------------------------
# Gaussian Blur
# (ลด noise ช่วยให้ edge เนียนขึ้น)
# --------------------------
blur = cv2.GaussianBlur(img, (5, 5), 1.4)

# --------------------------
# Canny edge detection
# --------------------------
edges = cv2.Canny(blur, 100, 200)

# --------------------------
# บันทึกไฟล์
# --------------------------
cv2.imwrite(output_path, edges)

print("✅ บันทึกไฟล์สำเร็จ →", output_path)
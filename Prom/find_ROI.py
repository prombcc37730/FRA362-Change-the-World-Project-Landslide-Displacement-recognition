import cv2

# โหลดภาพ (แทนที่ path ด้วยไฟล์ของคุณ)
img = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_1.jpg")   # ตัวอย่างไฟล์ที่คุณอัปโหลด

# ============================
#  ปรับค่า ROI ที่นี่
# ============================
x = 400   # ตำแหน่งซ้าย
y = 2200   # ตำแหน่งบน
w = 3000    # ความกว้าง ROI
h = 1000   # ความสูง ROI
# ============================

# วาดกรอบสี่เหลี่ยม ROI
img_draw = img.copy()
cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 3)

# แสดงผลแบบ resize เพื่อให้ไม่ใหญ่เกินไป
scale = 0.4   # ลดขนาดภาพลง 40%
img_resized = cv2.resize(img_draw, None, fx=scale, fy=scale)

cv2.imshow("Preview ROI", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
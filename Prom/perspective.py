import cv2
import numpy as np

# โหลดรูปภาพ
img_src = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_2.jpg") # รูปต้นฉบับที่คุณต้องการแปลง
img_dst = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_1.jpg") # รูปเป้าหมายที่คุณต้องการให้เหมือน

# 1. กำหนดจุดคู่กัน (อย่างน้อย 4 จุด)
# โดยทั่วไป เราจะเลือกจุดเหล่านี้ด้วยตนเอง หรือใช้ algorithm ตรวจจับคุณสมบัติ
# นี่คือตัวอย่างสมมติของจุด 4 จุด:
# format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

# จุดบนภาพต้นฉบับ (source image)
src_pts = np.float32([
    [100, 100],  # Top-left corner example
    [400, 100],  # Top-right
    [400, 300],  # Bottom-right
    [100, 300]   # Bottom-left
]).reshape(-1, 1, 2)

# จุดบนภาพเป้าหมาย (destination image) - นี่คือตำแหน่งที่คุณต้องการให้จุดจาก src_pts ไปอยู่
# สมมติว่าต้องการให้ไปอยู่ในพื้นที่สี่เหลี่ยมเบี้ยวในภาพเป้าหมาย
dst_pts = np.float32([
    [50, 150],   # New Top-left
    [450, 120],  # New Top-right
    [420, 350],  # New Bottom-right
    [80, 380]    # New Bottom-left
]).reshape(-1, 1, 2)


# 2. คำนวณ Homography Matrix (H)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# cv2.RANSAC เป็นวิธีที่ทนทานต่อ outliers (จุดที่ไม่ตรงกันดี)

# 3. ประยุกต์ใช้ Perspective Transformation
# กำหนดขนาดของภาพผลลัพธ์ (โดยปกติจะใช้ขนาดของภาพเป้าหมาย)
height, width, _ = img_dst.shape
img_transformed = cv2.warpPerspective(img_src, H, (width, height))

# แสดงผลลัพธ์
cv2.imshow("Source Image", img_src)
cv2.imshow("Destination Image", img_dst)
cv2.imwrite(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\perspective\per1_7.jpg", img_transformed)
cv2.imshow("Transformed Image", img_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

# -------------------------
# 1) โหลดภาพ Before / After
# -------------------------
img1 = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_1.jpg")   # ภาพฐาน (Reference)
img2 = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_7.jpg")    # ภาพที่ต้องการ align

# แปลงเป็น grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# -------------------------
# 2) สร้าง SIFT detector
# -------------------------
sift = cv2.SIFT_create()

# หา keypoints และ descriptors
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# -------------------------
# 3) FLANN matcher สำหรับจับคู่ feature
# -------------------------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des2, des1, k=2)   # หลัง → ก่อน

# -------------------------
# 4) Lowe's Ratio Test
# -------------------------
good = []
ratio_thresh = 0.7
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good.append(m)

print("Good matches:", len(good))

# ต้องมีอย่างน้อย 10 คู่ เพื่อคำนวณ Homography
if len(good) > 10:
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # -------------------------
    # 5) หา Homography Matrix
    # -------------------------
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # -------------------------
    # 6) Warp Image (Align)
    # -------------------------
    height, width, channels = img1.shape
    aligned_img = cv2.warpPerspective(img2, H, (width, height))

    # -------------------------
    # 7) บันทึกผลลัพธ์
    # -------------------------
    cv2.imwrite("aligned_output_3.jpg", aligned_img)
    cv2.imwrite("matches3.jpg", cv2.drawMatches(img2, kp2, img1, kp1, good, None))

    print("Alignment completed → saved as aligned_output_2.jpg")
else:
    print("Not enough matches found.")
import cv2
import numpy as np

# โหลดภาพ
img1 = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_1.jpg")
img2 = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L1_7.jpg")


# -----------------------------------------------------
# (1) เลือกกรอบ ROI ที่จะใช้หา feature
# -----------------------------------------------------
x = 400   # ตำแหน่งซ้าย
y = 2200   # ตำแหน่งบน
w = 3000    # ความกว้าง ROI
h = 1000   # ความสูง ROI

roi = img1[y:y+h, x:x+w]

# ----------------------------
# 3) สร้าง SIFT
# ----------------------------
sift = cv2.SIFT_create()

# Detect เฉพาะใน ROI
kp_roi, des_roi = sift.detectAndCompute(roi, None)

# แต่ตำแหน่ง keypoint ใน ROI ต้องถูกเลื่อนกลับไปอยู่ในภาพจริง
kp1 = []
for kp in kp_roi:
    new_kp = cv2.KeyPoint(kp.pt[0] + x, kp.pt[1] + y, kp.size, kp.angle,
                           kp.response, kp.octave, kp.class_id)
    kp1.append(new_kp)
des1 = des_roi

# Detect ทั้งภาพใน img2
kp2, des2 = sift.detectAndCompute(img2, None)

# ----------------------------
# 4) FLANN Matching
# ----------------------------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print("Good matches:", len(good))

# ต้องการอย่างน้อย 4 match
if len(good) >= 4:

    # ----------------------------
    # 5) ใช้ RANSAC หา Homography
    # ----------------------------
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    print("Homography:\n", H)

    # ----------------------------
    # 6) วาด ROI ที่ถูก warp ไปบนภาพ 2
    # ----------------------------
    pts = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    img2_draw = img2.copy()
    cv2.polylines(img2_draw, [np.int32(dst)], True, (0, 255, 0), 3)

# ----------------------------
    # 7) ทำการ Warping img2 ให้จัดแนวตาม img1
    # ----------------------------

    # 7.1) คำนวณ Inverse Homography (H_inv)
    # H_inv ใช้สำหรับแปลง img2 -> img1
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("คำนวณ Inverse Homography ล้มเหลว (Singular Matrix)")
        exit() # ออกจากโปรแกรมหากคำนวณไม่ได้

    # 7.2) กำหนดขนาดของภาพผลลัพธ์
    # ใช้ขนาดของ img1 เพื่อให้ img2 ที่ถูกปรับมีขนาดเท่ากัน
    height, width, _ = img1.shape

    # 7.3) ทำการ Warping
    # ปรับ img2 ให้จัดแนวตาม img1 โดยใช้ H_inv และขนาดของ img1
    aligned_img2 = cv2.warpPerspective(img2, H_inv, (width, height))

    # ----------------------------
    # 8) แสดงผลลัพธ์
    # ----------------------------

# บันทึกภาพที่จัดแนวแล้ว (aligned_img2)
    cv2.imwrite("aligned_result_img2.jpg", aligned_img2)
    print("ภาพที่จัดแนวแล้วถูกบันทึกที่: aligned_result_img2.jpg")
    
    # สร้างภาพเปรียบเทียบ
    comparison_image = np.hstack((img1, aligned_img2))

    # บันทึกภาพเปรียบเทียบ
    cv2.imwrite("comparison_aligned.jpg", comparison_image)
    print("ภาพเปรียบเทียบถูกบันทึกที่: comparison_aligned.jpg")

    # แสดงผลลัพธ์บนหน้าจอ
    comparison_resized = cv2.resize(comparison_image, None, fx=0.4, fy=0.4)
    cv2.imshow("Alignment Result (img1 | Aligned img2)", comparison_resized)

    # แสดงภาพ Matches
    draw_params = dict(matchColor=(0,255,0),
                         singlePointColor=None,
                         matchesMask=matchesMask,
                         flags=cv2.DrawMatchesFlags_DEFAULT)
    
    result_matches = cv2.drawMatches(img1, kp1, img2_draw, kp2, good, None, **draw_params)
    result_matches_resized = cv2.resize(result_matches, None, fx=0.4, fy=0.4)
    cv2.imshow("Matches + RANSAC ROI", result_matches_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Not enough good matches for RANSAC (Need >= 4).")
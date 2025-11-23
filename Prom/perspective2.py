import cv2
import numpy as np

# โหลดรูปภาพ
# ตรวจสอบให้แน่ใจว่า Path ถูกต้องและไฟล์มีอยู่จริง
img_src = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L3_2.jpg")
img_dst = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L3_1.jpg")

# แปลงเป็น Grayscale เพื่อใช้กับ SIFT
img1_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

# ตรวจสอบว่าภาพโหลดสำเร็จหรือไม่
if img_src is None or img_dst is None:
    print("❌ Error: Could not load one or both images.")
    exit()

# ใช้อัลกอริทึม SIFT ในการตรวจจับ Keypoints และ Descriptors
sift = cv2.SIFT_create()

# ตรวจจับ Keypoints (kp) และคำนวณ Descriptors (des)
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# ตรวจสอบว่า Descriptors มีอยู่จริง
if des1 is None or des2 is None:
    print("❌ Error: Could not find sufficient features in one or both images.")
    exit()


# กำหนดพารามิเตอร์สำหรับ FLANN
# Index parameters สำหรับ SIFT/SURF (ใช้ LSH_INDEX สำหรับ ORB/BRIEF)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

# Search parameters
search_params = dict(checks = 50)  

# สร้าง FLANN Matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# ใช้ k-NN Matcher (k=2) เพื่อหา 2 จุดใกล้เคียงที่สุด สำหรับการทดสอบอัตราส่วน
matches = flann.knnMatch(des1, des2, k=2)

# การกรองจุดคู่กันด้วย Ratio Test ของ David Lowe
good_matches = []
for m, n in matches:
    # หากจุดที่ใกล้ที่สุด (m) มีระยะห่างน้อยกว่า 70% ของจุดที่ใกล้ที่สุดรองลงมา (n) 
    # ถือว่าเป็นจุดคู่กันที่ดี (Ratio < 0.70)
    if m.distance < 0.75 * n.distance: 
        good_matches.append(m)

# กำหนดจำนวนจุดคู่กันขั้นต่ำที่ต้องการ (Min 4)
MIN_MATCH_COUNT = 10 

if len(good_matches) > MIN_MATCH_COUNT:
    # แยกพิกัดของจุดคู่กัน
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 2. คำนวณ Homography Matrix (H)
    # ใช้ RANSAC เพื่อกำจัด Outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 3. ประยุกต์ใช้ Perspective Transformation
    height, width, _ = img_dst.shape
    img_transformed = cv2.warpPerspective(img_src, H, (width, height))
    
    # บันทึกและแสดงผลลัพธ์
    output_path = r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\perspective\per_flann_result3_12.jpg"
    cv2.imwrite(output_path, img_transformed)
    
    print(f"✅ บันทึกภาพผลลัพธ์สำเร็จ ที่: {output_path}")

    # แสดงภาพ
    #cv2.imshow("Source Image", img_src)
    #cv2.imshow("Destination Image", img_dst)
    #cv2.imshow("Transformed Image (FLANN)", img_transformed)
    
    # (Optional) แสดงการจับคู่จุด
    draw_params = dict(matchColor = (0, 255, 0), # สีเขียวสำหรับ Matches
                       singlePointColor = None,
                       matchesMask = mask.ravel().tolist(), # ใช้ Mask จาก RANSAC
                       flags = 2)
    img_matches = cv2.drawMatches(img_src, kp1, img_dst, kp2, good_matches, None, **draw_params)
    #cv2.imshow("Good Matches (FLANN)", img_matches)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print(f"❌ ไม่พบจุดคู่กันที่เชื่อถือได้เพียงพอ - Found only {len(good_matches)}/{MIN_MATCH_COUNT} matches.")
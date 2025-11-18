import cv2
import numpy as np

# ---------------------------------------
# Load input images (grayscale)
# ---------------------------------------
imgA = cv2.imread("Foundation Dataset/Location 1/083142_0831_2025.JPG", cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread("Foundation Dataset/Location 1/101332_0831_2025.JPG", cv2.IMREAD_GRAYSCALE)

if imgA is None or imgB is None:
    raise ValueError("Error: Could not load one or both images.")

# ---------------------------------------
# Define ROI/frame in image A
# (Edit these values for your frame)
# ---------------------------------------
x, y, w, h = 496, 2320, 280, 236     # example ROI coordinates
roiA = imgA[y:y+h, x:x+w]

# ---------------------------------------
# Feature detection (SIFT)
# ---------------------------------------
sift = cv2.SIFT_create()

# Detect features only in ROI
kp_roi, des_roi = sift.detectAndCompute(roiA, None)

# Shift keypoints from ROI to image coordinates
kpA = []
for kp in kp_roi:
    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
    kpA.append(kp)

# Detect features in full image B
kpB, desB = sift.detectAndCompute(imgB, None)

# ---------------------------------------
# Feature Matching
# ---------------------------------------
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des_roi, desB, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:   # Lowe ratio test
        good.append(m)

if len(good) < 4:
    raise ValueError("Not enough good matches found to compute transformation.")

# ---------------------------------------
# Build corresponding point sets
# ---------------------------------------
src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# ---------------------------------------
# Compute Homography (Perspective)
# ---------------------------------------
H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

print("\n=====================================")
print(" Homography Matrix (3x3)")
print("=====================================")
print(H)

# ---------------------------------------
# Compute Affine (Similarity / Linear)
# ---------------------------------------
A, inliers = cv2.estimateAffine2D(dst_pts, src_pts)

print("\n=====================================")
print(" Affine Transform Matrix (2x3)")
print("=====================================")
print(A)

if A is not None:
    warpedB_affine = cv2.warpAffine(imgB, A, (imgA.shape[1], imgA.shape[0]))
    cv2.imwrite("warped_imageB_affine.jpg", warpedB_affine)

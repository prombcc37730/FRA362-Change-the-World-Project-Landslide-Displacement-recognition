import cv2

img = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\Foundation Dataset\Location 1\083142_0831_2025.jpg")

# แปลงเป็น YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Equalize เฉพาะช่อง Y (luminance)
ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])

# แปลงกลับ BGR
equalized_color = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imwrite(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\Equalization_image\Eq_083142_0831_2025.jpg", equalized_color)
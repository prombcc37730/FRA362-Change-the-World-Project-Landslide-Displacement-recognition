import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

img_transformed = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\perspective\per_flann_result3_12.jpg")
img_dst = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\FRA362 Dataset\L3_1.jpg")


img_trans_gray = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)
img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

diff = cv2.absdiff(img_trans_gray, img_dst_gray)
# ตั้งค่าเพื่อให้แสดงภาษาไทยได้ใน Matplotlib (ถ้าจำเป็น)
plt.rcParams['font.family'] = 'Tahoma' 
plt.rcParams['font.size'] = 10

# 2.1 แสดงผลต่าง (diff) เป็น Heatmap
fig, ax = plt.subplots(figsize=(8, 8))

# ใช้ plt.imshow เพื่อแสดงอาร์เรย์ NumPy เป็นภาพ
# 'cmap' คือ Colormap ที่ใช้ (เช่น 'jet', 'hot', 'viridis')
# 'interpolation' = 'nearest' เพื่อรักษาขอบเขตที่คมชัด
# 'extent' ใช้เพื่อให้แกนตรงกับขนาดภาพ
heatmap = ax.imshow(diff, cmap='jet', interpolation='nearest')

# 2.2 เพิ่ม Color Bar เพื่อแสดงมาตราส่วน
cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Absolute Pixel Difference (ความแตกต่างของพิกเซลสัมบูรณ์)', rotation=270, labelpad=15)

# 2.3 ตั้งชื่อและแสดงผล
ax.set_title('Heatmap of Perspective Transformation Difference')
plt.axis('off') # ปิดแกน x, y
plt.show()

# (Optional) บันทึก Heatmap เป็นไฟล์ภาพ
# plt.savefig(r"C:\Users\Win10\Documents\GitHub\...\per_heatmap_result.png", bbox_inches='tight')
import cv2
import numpy as np
import matplotlib.pyplot as plt

def perform_change_detection(image_before_path, image_after_path):
    """
    ทำการตรวจจับความแตกต่าง (Change Detection) โดยการลบภาพ Edge Detection สองภาพ
    
    Args:
        image_before_path (str): ชื่อไฟล์ภาพ 'ก่อน' การเปลี่ยนแปลง
        image_after_path (str): ชื่อไฟล์ภาพ 'หลัง' การเปลี่ยนแปลง
    
    Returns:
        np.ndarray: ภาพผลต่าง (Difference Image) ที่แสดงบริเวณที่มีการเปลี่ยนแปลง
    """
    
    # 1. โหลดภาพทั้งสองในโหมด Grayscale
    # ภาพ Edge Detection มักจะเป็น Grayscale อยู่แล้ว การโหลดแบบนี้ช่วยให้มั่นใจ
    img_before = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\edge_detection_image\canny_083142_0831_2025.jpg", cv2.IMREAD_GRAYSCALE)
    img_after = cv2.imread(r"C:\Users\Win10\Documents\GitHub\FRA362-Change-the-World-Project-Landslide-Displacement-recognition\Prom\edge_detection_image\canny_090432_0904_2025.jpg", cv2.IMREAD_GRAYSCALE)

    # ตรวจสอบว่าโหลดภาพสำเร็จหรือไม่
    if img_before is None:
        print(f"เกิดข้อผิดพลาด: ไม่สามารถโหลดภาพ 'ก่อน' ที่ {image_before_path} ได้")
        return None
    if img_after is None:
        print(f"เกิดข้อผิดพลาด: ไม่สามารถโหลดภาพ 'หลัง' ที่ {image_after_path} ได้")
        return None

    # 2. ตรวจสอบขนาดของภาพ (สำคัญมากสำหรับการลบภาพ)
    if img_before.shape != img_after.shape:
        print("คำเตือน: ขนาดของภาพไม่เท่ากัน! โปรดตรวจสอบการ Alignment/Registration ของภาพ")
        # ในทางปฏิบัติควรมีการทำ Image Registration แต่สำหรับการทดสอบเบื้องต้น
        # เราอาจปรับขนาด (Resize) ภาพหนึ่งให้เท่ากับอีกภาพ
        img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]), interpolation=cv2.INTER_LINEAR)
        print(f"ทำการปรับขนาดภาพ 'หลัง' เป็น {img_before.shape}")


    # 3. การลบภาพเพื่อหาความแตกต่าง (Absolute Difference)
    # cv2.absdiff จะคำนวณค่าสัมบูรณ์ของความแตกต่างของพิกเซลแต่ละคู่
    # Output = |Image_After - Image_Before|
    diff_image = cv2.absdiff(img_after, img_before)


    # 4. การปรับเกณฑ์ (Thresholding) เพื่อเน้นบริเวณที่เปลี่ยนแปลง
    # เราตั้งค่า Threshold เพื่อให้ความแตกต่างที่ชัดเจนเท่านั้นที่ปรากฏเป็นสีขาว
    # ret: ค่า Threshold ที่ใช้, thresh_img: ภาพที่ผ่านการ Threshold แล้ว
    # ค่า 30 เป็นค่าเริ่มต้น สามารถปรับเปลี่ยนได้
    threshold_value = 30 
    max_value = 255 # ค่าสูงสุดที่จะกำหนดให้กับพิกเซลที่ผ่านเกณฑ์
    
    _, thresh_diff = cv2.threshold(diff_image, threshold_value, max_value, cv2.THRESH_BINARY)
    
    
    # 5. แสดงผลลัพธ์
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_before, cmap='gray')
    plt.title('1. Image Before (ก่อน)')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_after, cmap='gray')
    plt.title('2. Image After (หลัง)')
    
    plt.subplot(1, 3, 3)
    plt.imshow(thresh_diff, cmap='gray')
    plt.title(f'3. Change Detected (|After - Before| > {threshold_value})')
    
    plt.show()

    # บันทึกภาพผลต่าง
    output_path = "change_detection_result.png"
    cv2.imwrite(output_path, thresh_diff)
    print(f"\nบันทึกภาพผลลัพธ์ที่: {output_path}")

    return thresh_diff

# --- ส่วนการใช้งานโค้ด ---
# **สำคัญ:** เปลี่ยนชื่อไฟล์ภาพให้เป็นชื่อที่คุณมี
image_before_file = 'canny_083142_0831_2025.jpg' # ภาพ 1 (ก่อน)
image_after_file = 'canny_090432_0904_2025.jpg'  # ภาพ 2 (หลัง)

result_image = perform_change_detection(image_before_file, image_after_file)
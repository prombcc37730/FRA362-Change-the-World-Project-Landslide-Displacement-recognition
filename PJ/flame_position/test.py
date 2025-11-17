import cv2

SCALE = 0.25   #ไม่ปรับแล้วแม้งซูม

# ตัวแปร global
img_full = None  
img_show = None     
drawing = False

x0_show, y0_show = -1, -1
x1_show, y1_show = -1, -1
roi_corners_real = None


def mouse_rect(event, x, y, flags, param):
    global drawing, x0_show, y0_show, x1_show, y1_show
    global img_show, roi_corners_real, img_full

    # ---------- START DRAG ----------
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0_show, y0_show = x, y
        x1_show, y1_show = x, y
        img_show = cv2.resize(img_full, None, fx=SCALE, fy=SCALE)

    # ---------- DRAGGING ----------
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x1_show, y1_show = x, y
        img_show = cv2.resize(img_full, None, fx=SCALE, fy=SCALE)

        
        cv2.rectangle(img_show, (x0_show, y0_show),
                      (x1_show, y1_show), (0, 0, 255), 2)



    # ---------- FINISH DRAG ----------
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1_show, y1_show = x, y

  
        img_show = cv2.resize(img_full, None, fx=SCALE, fy=SCALE)

        x_min_show, x_max_show = sorted([x0_show, x1_show])
        y_min_show, y_max_show = sorted([y0_show, y1_show])

        cv2.rectangle(img_show, (x_min_show, y_min_show),
                      (x_max_show, y_max_show), (0, 0, 255), 2)

        # แปลงพิกัดกลับ ไปเป็นของภาพจริง
        x_min_real = int(x_min_show / SCALE)
        y_min_real = int(y_min_show / SCALE)
        x_max_real = int(x_max_show / SCALE)
        y_max_real = int(y_max_show / SCALE)

        roi_corners_real = [
            (x_min_real, y_min_real),  # top-left
            (x_max_real, y_min_real),  # top-right
            (x_max_real, y_max_real),  # bottom-right
            (x_min_real, y_max_real),  # bottom-left
        ]

        print("\n[REAL IMAGE] ROI corners (x, y):")
        for i, (xr, yr) in enumerate(roi_corners_real, 1):
            print(f"  P{i}: ({xr}, {yr})")


def main():
    global img_full, img_show

    # โหลดภาพ
    img_full = cv2.imread("A.JPG")

    # แสดงผล
    img_show = cv2.resize(img_full, None, fx=SCALE, fy=SCALE)

    cv2.namedWindow("Image A", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image A", mouse_rect)

    print("กด q หรือ ESC ออก")

    while True:
        cv2.imshow("Image A", img_show)
        key = cv2.waitKey(20)
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

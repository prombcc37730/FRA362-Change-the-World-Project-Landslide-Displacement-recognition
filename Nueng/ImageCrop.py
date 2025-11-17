import cv2

# This code use Claude AI to optimize only not generate at all.

path = "Foundation Dataset/Location 1/083142_0831_2025.JPG"
scale = 0.3

# State variables
drawing = False
x0, y0 = -1, -1
x1, y1 = -1, -1
cropped_image = None
base_image = None  # Store the base resized image


def read_image(image_path):
    """Read image from file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


def resize_image(image, scale):
    """Resize image by scale factor."""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for cropping."""
    global drawing, x0, y0, x1, y1, cropped_image, base_image
    
    image, scale = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
        x1, y1 = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x1, y1 = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = x, y
        
        # Convert to original image coordinates
        orig_x0 = int(x0 / scale)
        orig_y0 = int(y0 / scale)
        orig_x1 = int(x1 / scale)
        orig_y1 = int(y1 / scale)
        
        # Ensure coordinates are in correct order
        orig_x0, orig_x1 = min(orig_x0, orig_x1), max(orig_x0, orig_x1)
        orig_y0, orig_y1 = min(orig_y0, orig_y1), max(orig_y0, orig_y1)
        
        print(f"Cropped coordinates: ({orig_x0}, {orig_y0}) to ({orig_x1}, {orig_y1})")
        
        # Store cropped image
        cropped_image = image[orig_y0:orig_y1, orig_x0:orig_x1].copy()


def main():
    """Main function to run the cropping tool."""
    global cropped_image, base_image
    
    # Read and resize image
    image = read_image(path)
    base_image = resize_image(image, scale)
    
    # Create window and set callback
    cv2.namedWindow("Image Cropper")
    cv2.setMouseCallback("Image Cropper", mouse_callback, (image, scale))
    
    print("Instructions:")
    print("- Click and drag to select area")
    print("- Press 'q' or ESC to finish")
    print("- Selected area will be shown after closing")
    
    # Main loop - draw here instead of in callback
    while True:
        # Start with clean base image
        display = base_image.copy()
        
        # Draw rectangle if currently drawing or if we have a selection
        if drawing or (x0 != -1 and x1 != -1):
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 0, 255), 2)
        
        cv2.imshow("Image Cropper", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    cv2.destroyAllWindows()
    
    # Show cropped image if available
    if cropped_image is not None and cropped_image.size > 0:
        cv2.imshow("Cropped Image", cropped_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No crop area selected.")


if __name__ == "__main__":
    main()
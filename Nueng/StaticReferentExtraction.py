import cv2
import numpy as np


def crop_images_interactive(image_paths, scale=0.3, show_result=True, upscale_factor=1.0):
    """
    Interactive image cropping tool for multiple images.
    Switch between images and crop the selected one.
    
    Args:
        image_paths (list): List of image file paths
        scale (float): Scale factor for display (default: 0.3)
        show_result (bool): Whether to show the cropped result (default: True)
        upscale_factor (float): Factor to upscale cropped image (default: 1.0, no upscaling)
                               Use 2.0 for 2x resolution, etc.
    
    Returns:
        tuple: (cropped_image, coordinates, image_index) 
               where coordinates is ((x0, y0), (x1, y1))
               Returns (None, None, None) if no selection was made
    
    Controls:
        - Click and drag to select area
        - Press '1', '2', '3'... to switch between images
        - Press 'q' or ESC to finish
    
    Usage:
        paths = ["image1.jpg", "image2.jpg"]
        cropped, coords, idx = crop_images_interactive(paths, upscale_factor=2.0)
    """
    
    if not image_paths:
        raise ValueError("image_paths list cannot be empty")
    
    # Load all images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load {path}")
        else:
            images.append((img, path))
    
    if not images:
        raise FileNotFoundError("No valid images found")
    
    # State variables
    state = {
        'drawing': False,
        'x0': -1, 'y0': -1,
        'x1': -1, 'y1': -1,
        'cropped': None,
        'coords': None,
        'current_index': 0,
        'base_images': []
    }
    
    # Precompute resized images
    for img, path in images:
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        state['base_images'].append(resized)
    
    def mouse_callback(event, x, y, flags, param):
        """Handle mouse events for cropping."""
        if event == cv2.EVENT_LBUTTONDOWN:
            state['drawing'] = True
            state['x0'], state['y0'] = x, y
            state['x1'], state['y1'] = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if state['drawing']:
                state['x1'], state['y1'] = x, y
            
        elif event == cv2.EVENT_LBUTTONUP:
            state['drawing'] = False
            state['x1'], state['y1'] = x, y
            
            # Get current image
            current_image = images[state['current_index']][0]
            
            # Convert to original image coordinates
            orig_x0 = int(state['x0'] / scale)
            orig_y0 = int(state['y0'] / scale)
            orig_x1 = int(state['x1'] / scale)
            orig_y1 = int(state['y1'] / scale)
            
            # Ensure coordinates are in correct order
            orig_x0, orig_x1 = min(orig_x0, orig_x1), max(orig_x0, orig_x1)
            orig_y0, orig_y1 = min(orig_y0, orig_y1), max(orig_y0, orig_y1)
            
            state['coords'] = ((orig_x0, orig_y0), (orig_x1, orig_y1))
            print(f"Selected coordinates on image {state['current_index'] + 1}: ({orig_x0}, {orig_y0}) to ({orig_x1}, {orig_y1})")
            
            # Store cropped image
            if orig_x1 > orig_x0 and orig_y1 > orig_y0:
                cropped = current_image[orig_y0:orig_y1, orig_x0:orig_x1].copy()
                
                # Apply upscaling if requested
                if upscale_factor > 1.0:
                    new_width = int(cropped.shape[1] * upscale_factor)
                    new_height = int(cropped.shape[0] * upscale_factor)
                    # Use INTER_CUBIC for better quality upscaling
                    cropped = cv2.resize(cropped, (new_width, new_height), 
                                        interpolation=cv2.INTER_CUBIC)
                    print(f"Upscaled to {new_width}x{new_height} (factor: {upscale_factor}x)")
                
                state['cropped'] = cropped
    
    # Create window and set callback
    window_name = "Image Cropper - Multi-Image"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Instructions:")
    print(f"- Loaded {len(images)} images")
    print("- Press '1', '2', '3'... to switch between images")
    print("- Click and drag to select area to crop")
    print("- Press 'q' or ESC to finish and crop selected area")
    print()
    
    # Main loop
    while True:
        # Get current base image
        base_image = state['base_images'][state['current_index']]
        display = base_image.copy()
        
        # Draw rectangle if currently drawing or if we have a selection
        if state['drawing'] or (state['x0'] != -1 and state['x1'] != -1):
            cv2.rectangle(display, 
                         (state['x0'], state['y0']), 
                         (state['x1'], state['y1']), 
                         (0, 0, 255), 2)
        
        # Add image indicator
        current_path = images[state['current_index']][1]
        text = f"Image {state['current_index'] + 1}/{len(images)}: {current_path.split('/')[-1]}"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Switch between images with number keys
        if key >= ord('1') and key <= ord('9'):
            new_index = key - ord('1')
            if new_index < len(images):
                state['current_index'] = new_index
                # Reset selection when switching images
                state['x0'], state['y0'] = -1, -1
                state['x1'], state['y1'] = -1, -1
                state['drawing'] = False
                print(f"Switched to image {new_index + 1}: {images[new_index][1]}")
        
        # Exit
        elif key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    cv2.destroyAllWindows()
    
    # Show cropped result if requested
    if show_result and state['cropped'] is not None and state['cropped'].size > 0:
        cv2.imshow("Cropped Image", state['cropped'])
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return state['cropped'], state['coords'], state['current_index']


# Example usage
if __name__ == "__main__":
    # Multiple images
    paths = [
        "Foundation Dataset/Location 1/083142_0831_2025.JPG",
        "Foundation Dataset/Location 1/090432_0904_2025.JPG"  # Add your second image path
    ]
    
    # Get cropped image, coordinates, and which image was selected
    cropped, coords, image_idx = crop_images_interactive(paths, scale=0.3, upscale_factor=2.0)
    
    # Use the results
    if cropped is not None:
        print(f"\nCrop successful from image {image_idx + 1}!")
        print(f"Coordinates: {coords}")
        print(f"Cropped image size: {cropped.shape[1]}x{cropped.shape[0]}")
        print(f"Source: {paths[image_idx]}")
        # Optional: save the cropped image
        cv2.imwrite(f"cropped_from_image_{image_idx + 1}.jpg", cropped)
    else:
        print("No crop area selected.")
import cv2
import numpy as np


def crop_images_multi_select(image_paths, scale=0.3, show_results=True):
    """
    Interactive image cropping tool with multiple selection support.
    Make multiple crops across different images and get all coordinates.
    
    Args:
        image_paths (list): List of image file paths
        scale (float): Scale factor for display (default: 0.3)
        show_results (bool): Whether to show the cropped results (default: True)
    
    Returns:
        list of dict: Each dict contains:
            - 'image_index': Index of the source image (0-based)
            - 'image_path': Path to the source image
            - 'x': Top-left x coordinate in original image
            - 'y': Top-left y coordinate in original image
            - 'w': Width of the crop
            - 'h': Height of the crop
            - 'cropped_image': The cropped image array
            - 'selection_order': Order in which this crop was made
        Returns empty list if no selections were made
    
    Controls:
        - Click and drag to select area
        - Press 'c' to confirm and save current selection
        - Press '1', '2', '3'... to switch between images
        - Press 'd' to delete last selection
        - Press 'r' to reset current rectangle (before confirming)
        - Press 'q' or ESC to finish and return all selections
    
    Usage:
        paths = ["image1.jpg", "image2.jpg"]
        selections = crop_images_multi_select(paths, scale=0.3)
        
        for sel in selections:
            print(f"Image {sel['image_index']}: x={sel['x']}, y={sel['y']}, "
                  f"w={sel['w']}, h={sel['h']}")
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
        'current_index': 0,
        'base_images': [],
        'selections': [],  # List to store all confirmed selections
        'temp_rect': None  # Current rectangle being drawn
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
            
            # Store temporary rectangle
            if abs(state['x1'] - state['x0']) > 5 and abs(state['y1'] - state['y0']) > 5:
                state['temp_rect'] = {
                    'x0': state['x0'],
                    'y0': state['y0'],
                    'x1': state['x1'],
                    'y1': state['y1'],
                    'image_index': state['current_index']
                }
                print(f"Rectangle drawn. Press 'c' to confirm or 'r' to reset")
    
    def confirm_selection():
        """Confirm and save the current selection."""
        if state['temp_rect'] is None:
            print("No rectangle to confirm!")
            return
        
        rect = state['temp_rect']
        current_image = images[rect['image_index']][0]
        current_path = images[rect['image_index']][1]
        
        # Convert to original image coordinates
        orig_x0 = int(rect['x0'] / scale)
        orig_y0 = int(rect['y0'] / scale)
        orig_x1 = int(rect['x1'] / scale)
        orig_y1 = int(rect['y1'] / scale)
        
        # Ensure coordinates are in correct order
        orig_x0, orig_x1 = min(orig_x0, orig_x1), max(orig_x0, orig_x1)
        orig_y0, orig_y1 = min(orig_y0, orig_y1), max(orig_y0, orig_y1)
        
        # Clamp to image boundaries
        orig_x0 = max(0, min(orig_x0, current_image.shape[1] - 1))
        orig_x1 = max(0, min(orig_x1, current_image.shape[1]))
        orig_y0 = max(0, min(orig_y0, current_image.shape[0] - 1))
        orig_y1 = max(0, min(orig_y1, current_image.shape[0]))
        
        # Calculate width and height
        w = orig_x1 - orig_x0
        h = orig_y1 - orig_y0
        
        # Minimum size check
        if w < 10 or h < 10:
            print("Selection too small! Minimum size is 10x10 pixels.")
            return
        
        # Crop the image
        cropped = current_image[orig_y0:orig_y1, orig_x0:orig_x1].copy()
        
        # Store selection
        selection = {
            'image_index': rect['image_index'],
            'image_path': current_path,
            'x': orig_x0,
            'y': orig_y0,
            'w': w,
            'h': h,
            'cropped_image': cropped,
            'selection_order': len(state['selections']) + 1,
            # Store scaled coordinates for visualization
            '_scaled_x0': rect['x0'],
            '_scaled_y0': rect['y0'],
            '_scaled_x1': rect['x1'],
            '_scaled_y1': rect['y1']
        }
        
        state['selections'].append(selection)
        print(f"✓ Selection {len(state['selections'])} confirmed: "
              f"Image {rect['image_index'] + 1}, "
              f"x={orig_x0}, y={orig_y0}, w={w}, h={h}")
        
        # Reset temporary rectangle
        state['temp_rect'] = None
        state['x0'], state['y0'] = -1, -1
        state['x1'], state['y1'] = -1, -1
    
    def delete_last_selection():
        """Delete the most recent selection."""
        if state['selections']:
            removed = state['selections'].pop()
            print(f"✗ Deleted selection {removed['selection_order']} from image {removed['image_index'] + 1}")
        else:
            print("No selections to delete!")
    
    # Create window and set callback
    window_name = "Multi-Select Image Cropper"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("=" * 70)
    print("MULTI-SELECT IMAGE CROPPER")
    print("=" * 70)
    print(f"Loaded {len(images)} images")
    print("\nControls:")
    print("  - Click and drag to draw selection rectangle")
    print("  - Press 'c' to CONFIRM and save current selection")
    print("  - Press 'r' to RESET current rectangle (before confirming)")
    print("  - Press '1', '2', '3'... to switch between images")
    print("  - Press 'd' to DELETE last confirmed selection")
    print("  - Press 'q' or ESC to FINISH and return all selections")
    print("=" * 70)
    print()
    
    # Main loop
    while True:
        # Get current base image
        base_image = state['base_images'][state['current_index']].copy()
        display = base_image.copy()
        
        # Draw all confirmed selections for current image (in green)
        for sel in state['selections']:
            if sel['image_index'] == state['current_index']:
                cv2.rectangle(display,
                            (sel['_scaled_x0'], sel['_scaled_y0']),
                            (sel['_scaled_x1'], sel['_scaled_y1']),
                            (0, 255, 0), 2)
                # Add selection number
                cv2.putText(display, f"#{sel['selection_order']}", 
                           (sel['_scaled_x0'] + 5, sel['_scaled_y0'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw temporary rectangle (in yellow if exists, red if currently drawing)
        if state['temp_rect'] and state['temp_rect']['image_index'] == state['current_index']:
            rect = state['temp_rect']
            cv2.rectangle(display,
                         (rect['x0'], rect['y0']),
                         (rect['x1'], rect['y1']),
                         (0, 255, 255), 2)
            cv2.putText(display, "Press 'c' to confirm", 
                       (rect['x0'], rect['y0'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        elif state['drawing']:
            cv2.rectangle(display,
                         (state['x0'], state['y0']),
                         (state['x1'], state['y1']),
                         (0, 0, 255), 2)
        
        # Add information overlay
        current_path = images[state['current_index']][1]
        filename = current_path.split('/')[-1]
        
        # Image info
        cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
        cv2.rectangle(display, (0, 0), (display.shape[1], 80), (255, 255, 255), 2)
        
        text = f"Image {state['current_index'] + 1}/{len(images)}: {filename}"
        cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        sel_count = sum(1 for s in state['selections'] if s['image_index'] == state['current_index'])
        text2 = f"Selections on this image: {sel_count} | Total: {len(state['selections'])}"
        cv2.putText(display, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Confirm selection
        if key == ord('c'):
            confirm_selection()
        
        # Reset current rectangle
        elif key == ord('r'):
            if state['temp_rect']:
                state['temp_rect'] = None
                state['x0'], state['y0'] = -1, -1
                state['x1'], state['y1'] = -1, -1
                print("Rectangle reset")
            else:
                print("No rectangle to reset!")
        
        # Delete last selection
        elif key == ord('d'):
            delete_last_selection()
        
        # Switch between images with number keys
        elif key >= ord('1') and key <= ord('9'):
            new_index = key - ord('1')
            if new_index < len(images):
                state['current_index'] = new_index
                # Reset current drawing
                state['x0'], state['y0'] = -1, -1
                state['x1'], state['y1'] = -1, -1
                state['drawing'] = False
                state['temp_rect'] = None
                print(f"→ Switched to image {new_index + 1}: {images[new_index][1]}")
        
        # Exit
        elif key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
    cv2.destroyAllWindows()
    
    # Show results if requested
    if show_results and state['selections']:
        print(f"\n{'=' * 70}")
        print(f"SHOWING {len(state['selections'])} CROPPED SELECTIONS")
        print(f"{'=' * 70}")
        
        for i, sel in enumerate(state['selections']):
            window_name = f"Selection #{sel['selection_order']} - Image {sel['image_index'] + 1}"
            cv2.imshow(window_name, sel['cropped_image'])
            print(f"#{sel['selection_order']}: Image {sel['image_index'] + 1}, "
                  f"x={sel['x']}, y={sel['y']}, w={sel['w']}, h={sel['h']}")
        
        print("\nPress any key to close all windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Clean up the internal scaled coordinates before returning
    for sel in state['selections']:
        del sel['_scaled_x0']
        del sel['_scaled_y0']
        del sel['_scaled_x1']
        del sel['_scaled_y1']
    
    return state['selections']


# Example usage
if __name__ == "__main__":
    # Multiple images
    paths = [
        "Foundation Dataset/Location 1/083142_0831_2025.JPG",
        "Foundation Dataset/Location 1/132328_0831_2025.JPG"
    ]
    
    # Get all selections
    selections = crop_images_multi_select(paths, scale=0.3, show_results=True)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Total selections made: {len(selections)}\n")
    
    if selections:
        for sel in selections:
            print(f"Selection #{sel['selection_order']}:")
            print(f"  Image Index: {sel['image_index']}")
            print(f"  Image Path: {sel['image_path']}")
            print(f"  Position: x={sel['x']}, y={sel['y']}")
            print(f"  Size: w={sel['w']}, h={sel['h']}")
            print(f"  Cropped shape: {sel['cropped_image'].shape}")
            print()
    else:
        print("No selections were made.")
        
        
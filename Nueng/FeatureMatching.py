import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# ============= CONFIGURATION CONSTANTS =============
DEFAULT_RATIO_THRESH = 0.65
DEFAULT_RANSAC_THRESH = 5.0
MIN_MATCHES_DEFAULT = 15
DIFF_THRESHOLD = 30
SIFT_NFEATURES = 8000
ORB_NFEATURES = 10000
MAX_DISPLAY_MATCHES = 50

# Homography validation thresholds
MAX_SCALE_CHANGE = 3.0
MAX_SHEAR = 0.5
MIN_DETERMINANT = 0.1


def preprocess_image(img, enhance_contrast=True, denoise=True):
    """
    Preprocess for better feature detection in natural scenes
    
    Natural scenes often have:
    - Low contrast areas (fog, shadows)
    - Noise from vegetation movement
    - Varying illumination
    
    Args:
        img: Input image (BGR or grayscale)
        enhance_contrast: Apply CLAHE for local contrast enhancement
        denoise: Apply denoising filter
    
    Returns:
        Preprocessed grayscale image
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Denoise to reduce vegetation/noise artifacts
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # CLAHE for local contrast enhancement (helps with shadows/lighting)
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        return gray
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def create_stable_region_mask(img, focus_on_rocky=True):
    """
    Create mask to focus on stable regions (rocks, man-made objects)
    and ignore vegetation areas that change frequently
    
    For landslide monitoring, rocky areas and exposed earth are more stable
    
    Args:
        img: Input image
        focus_on_rocky: If True, focus on high-texture regions
    
    Returns:
        Binary mask (255 for stable regions, 0 otherwise)
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        if focus_on_rocky:
            # High texture variance = rocks, low variance = vegetation/sky
            # Compute local standard deviation (FIXED: proper float handling)
            kernel_size = 15
            gray_float = gray.astype(np.float32)
            
            mean = cv2.blur(gray_float, (kernel_size, kernel_size))
            mean_sq = cv2.blur(gray_float ** 2, (kernel_size, kernel_size))
            variance = np.maximum(mean_sq - mean ** 2, 0)
            std_dev = np.sqrt(variance)
            
            # Normalize for consistent thresholding
            std_dev_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Threshold to get high-texture regions
            _, mask = cv2.threshold(std_dev_norm, 20, 255, cv2.THRESH_BINARY)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        else:
            # Use entire image
            mask = np.ones(gray.shape, dtype=np.uint8) * 255
        
        return mask
    except Exception as e:
        print(f"Error creating mask: {e}")
        return None


def is_homography_valid(M, max_scale=MAX_SCALE_CHANGE, max_shear=MAX_SHEAR):
    """
    Validate homography matrix for reasonable transformations
    
    Args:
        M: 3x3 homography matrix
        max_scale: Maximum allowed scale change
        max_shear: Maximum allowed shear
    
    Returns:
        Boolean indicating if homography is valid
    """
    if M is None:
        return False
    
    try:
        # Check determinant for scale (should be close to 1 for similar images)
        det = abs(np.linalg.det(M[:2, :2]))
        if det < MIN_DETERMINANT or det < 1/max_scale or det > max_scale:
            print(f"Warning: Invalid scale change (det={det:.3f})")
            return False
        
        # Check for excessive perspective distortion
        if abs(M[2, 0]) > 0.001 or abs(M[2, 1]) > 0.001:
            print(f"Warning: Excessive perspective distortion")
            # Allow but warn - don't reject entirely
        
        # Check if transformation is too extreme
        if not np.all(np.isfinite(M)):
            print("Warning: Non-finite values in homography")
            return False
        
        return True
    except Exception as e:
        print(f"Error validating homography: {e}")
        return False


def find_homography(img1_ref, img2_to_align, method='sift', ratio_thresh=DEFAULT_RATIO_THRESH, 
                    ransac_thresh=DEFAULT_RANSAC_THRESH, min_matches=MIN_MATCHES_DEFAULT, 
                    use_mask=False):
    """
    Find homography matrix to align img2 to img1 (optimized for natural scenes)
    
    Args:
        img1_ref: Reference image (static/template)
        img2_to_align: Image to be aligned
        method: 'sift', 'orb', or 'akaze'
        ratio_thresh: Lowe's ratio test threshold (0.6-0.7 recommended)
        ransac_thresh: RANSAC reprojection threshold in pixels
        min_matches: Minimum number of matches required
        use_mask: Apply mask to focus on stable regions
    
    Returns:
        M: Homography matrix (3x3) or None
        matches_info: Dictionary with matching statistics or None
    """
    try:
        # Preprocess images
        gray1 = preprocess_image(img1_ref, enhance_contrast=True, denoise=True)
        gray2 = preprocess_image(img2_to_align, enhance_contrast=True, denoise=True)
        
        if gray1 is None or gray2 is None:
            print("Error: Preprocessing failed")
            return None, None
        
        # Create mask to focus on stable regions (optional)
        mask1 = create_stable_region_mask(img1_ref, focus_on_rocky=use_mask) if use_mask else None
        mask2 = create_stable_region_mask(img2_to_align, focus_on_rocky=use_mask) if use_mask else None
        
        # Choose detector (optimized for natural scenes)
        if method == 'sift':
            detector = cv2.SIFT_create(
                nfeatures=SIFT_NFEATURES,
                contrastThreshold=0.03,
                edgeThreshold=10,
                sigma=1.6
            )
            norm = cv2.NORM_L2
        elif method == 'orb':
            detector = cv2.ORB_create(
                nfeatures=ORB_NFEATURES,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                patchSize=31
            )
            norm = cv2.NORM_HAMMING
        elif method == 'akaze':
            detector = cv2.AKAZE_create(
                threshold=0.0001,
                nOctaves=4,
                nOctaveLayers=4
            )
            norm = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sift', 'orb', or 'akaze'")
        
        # Detect and compute with optional masks
        kp1, des1 = detector.detectAndCompute(gray1, mask1)
        kp2, des2 = detector.detectAndCompute(gray2, mask2)
        
        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            print("Error: Could not detect features")
            return None, None
        
        print(f"Keypoints - Reference: {len(kp1)}, To Align: {len(kp2)}")
        
        # Match features
        bf = cv2.BFMatcher(norm, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good.append(m)
        
        print(f"Matches after ratio test: {len(good)}")
        
        if len(good) < min_matches:
            print(f"Error: Not enough good matches (got {len(good)}, need {min_matches})")
            return None, None
        
        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
        
        if M is None:
            print("Error: Could not compute homography")
            return None, None
        
        # Validate homography
        if not is_homography_valid(M):
            print("Warning: Homography failed validation (may be unreliable)")
            # Continue but warn user
        
        # Calculate inliers
        matches_mask = mask.ravel().tolist()
        inliers = sum(matches_mask)
        inlier_ratio = inliers / len(good) if len(good) > 0 else 0
        
        matches_info = {
            'total_matches': len(good),
            'inliers': inliers,
            'outliers': len(good) - inliers,
            'inlier_ratio': inlier_ratio,
            'keypoints': (kp1, kp2),
            'good_matches': good,
            'mask': matches_mask
        }
        
        print(f"RANSAC - Inliers: {inliers}, Outliers: {len(good) - inliers}, "
              f"Ratio: {inlier_ratio:.2%}")
        
        return M, matches_info
    
    except Exception as e:
        print(f"Error in find_homography: {e}")
        return None, None


def align_image_no_crop(img_to_align, M, reference_shape):
    """
    Apply homography transformation WITHOUT cropping - preserves full transformed image
    
    Args:
        img_to_align: Image to transform
        M: Homography matrix from find_homography()
        reference_shape: Shape of reference image (height, width) - used for output size calculation
    
    Returns:
        Transformed image with full content preserved (no cropping)
    """
    try:
        h2, w2 = img_to_align.shape[:2]
        h1, w1 = reference_shape[:2]
        
        # Get corners of the image to be transformed
        corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        # Transform corners to see where they land in reference space
        transformed_corners = cv2.perspectiveTransform(corners_img2, M)
        
        # Also include reference image corners
        corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        
        # Combine all corners to find bounding box
        all_corners = np.concatenate([transformed_corners, corners_img1], axis=0)
        
        # Find min/max to get bounding box
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Calculate translation to shift everything to positive coordinates
        translation = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]])
        
        # Compute final transformation matrix
        M_translated = translation @ M
        
        # Calculate output size
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        print(f"Output size (no crop): {output_width}x{output_height}")
        
        # Apply transformation with full output size
        aligned_img = cv2.warpPerspective(img_to_align, M_translated, 
                                         (output_width, output_height))
        
        return aligned_img
    except Exception as e:
        print(f"Error aligning image: {e}")
        return None


def align_image(img_to_align, M, reference_shape):
    """
    Apply homography transformation to align image (CROPPED version)
    
    Args:
        img_to_align: Image to transform
        M: Homography matrix from find_homography()
        reference_shape: Shape of reference image (height, width)
    
    Returns:
        Transformed image aligned to reference (cropped to reference size)
    """
    try:
        h, w = reference_shape[:2]
        aligned_img = cv2.warpPerspective(img_to_align, M, (w, h))
        return aligned_img
    except Exception as e:
        print(f"Error aligning image: {e}")
        return None


def visualize_matches(img1, img2, matches_info, max_display=MAX_DISPLAY_MATCHES):
    """
    Visualize feature matches
    
    Args:
        img1: Reference image
        img2: Image to align
        matches_info: Dictionary from find_homography()
        max_display: Maximum number of matches to display
    
    Returns:
        Image showing feature matches or None
    """
    try:
        kp1, kp2 = matches_info['keypoints']
        good = matches_info['good_matches']
        mask = matches_info['mask']
        
        # Draw only inliers
        inlier_matches = [good[i] for i in range(len(good)) if mask[i]]
        
        # Limit display for clarity
        display_matches = inlier_matches[:max_display]
        
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, display_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return img_matches
    except Exception as e:
        print(f"Error visualizing matches: {e}")
        return None


def visualize_alignment(img_ref, img_original, img_aligned):
    """Visualize original and aligned images side by side"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert BGR to RGB for matplotlib
        img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_aligned_rgb = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(img_ref_rgb)
        axes[0].set_title('Reference Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(img_original_rgb)
        axes[1].set_title('Original Image (To Align)', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(img_aligned_rgb)
        axes[2].set_title('Aligned Image (No Crop)', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in visualization: {e}")


def create_difference_map(img_ref, img_aligned, threshold=DIFF_THRESHOLD):
    """
    Create difference map to visualize changes (useful for landslide detection)
    
    Args:
        img_ref: Reference image
        img_aligned: Aligned image
        threshold: Threshold for significant changes
    
    Returns:
        Tuple of (diff, thresh, diff_colored) or (None, None, None)
    """
    try:
        # Convert to grayscale
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference
        diff = cv2.absdiff(gray_ref, gray_aligned)
        
        # Apply threshold to highlight significant changes
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Create color-coded difference map
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        return diff, thresh, diff_colored
    except Exception as e:
        print(f"Error creating difference map: {e}")
        return None, None, None


def main(reference_path, template_path, output_path="aligned_image.jpg", 
         method='sift', use_mask=False, no_crop=True):
    """
    Main pipeline for image alignment
    
    Args:
        reference_path: Path to reference image
        template_path: Path to image to be aligned
        output_path: Path to save aligned image
        method: Feature detection method ('sift', 'orb', 'akaze')
        use_mask: Whether to use stable region masking
        no_crop: If True, preserve full transformed image without cropping
    """
    print("="*70)
    print("IMAGE ALIGNMENT FOR NATURAL SCENES - LANDSLIDE MONITORING")
    print("="*70)
    
    # Load images
    print(f"\nLoading images...")
    reference_img = cv2.imread(reference_path)
    template_img = cv2.imread(template_path)
    
    if reference_img is None:
        print(f"Error: Could not load reference image from '{reference_path}'")
        return False
    
    if template_img is None:
        print(f"Error: Could not load template image from '{template_path}'")
        return False
    
    print(f"Reference shape: {reference_img.shape}")
    print(f"Template shape: {template_img.shape}")
    print(f"Mode: {'NO CROP (Full transform)' if no_crop else 'CROPPED (Reference size)'}")
    print("\n" + "="*70)
    
    # Try different methods and choose the best
    best_method = None
    best_M = None
    best_info = None
    best_inlier_ratio = 0
    
    # For natural scenes: SIFT usually works best, then AKAZE, then ORB
    if method == 'auto':
        methods_to_try = ['sift', 'akaze', 'orb']
    else:
        methods_to_try = [method]
    
    for current_method in methods_to_try:
        print(f"\nTrying {current_method.upper()}...")
        print("-" * 70)
        
        M, matches_info = find_homography(
            reference_img, 
            template_img, 
            method=current_method,
            ratio_thresh=DEFAULT_RATIO_THRESH,
            ransac_thresh=DEFAULT_RANSAC_THRESH,
            min_matches=20,
            use_mask=use_mask
        )
        
        if M is not None and matches_info['inlier_ratio'] > best_inlier_ratio:
            best_inlier_ratio = matches_info['inlier_ratio']
            best_method = current_method
            best_M = M
            best_info = matches_info
    
    if best_M is None:
        print("\n" + "="*70)
        print("ERROR: Could not find valid homography with any method")
        print("="*70)
        print("\nTroubleshooting suggestions:")
        print("1. Ensure images overlap significantly (>30%)")
        print("2. Try using --use-mask flag for rocky terrain")
        print("3. Check if images are too different (lighting, season, angle)")
        print("4. Reduce --min-matches parameter")
        return False
    
    print("\n" + "="*70)
    print(f"✓ Best method: {best_method.upper()}")
    print(f"✓ Inlier ratio: {best_inlier_ratio:.2%}")
    print(f"✓ Inliers: {best_info['inliers']}/{best_info['total_matches']}")
    print("="*70)
    
    # Apply transformation
    print("\nApplying transformation...")
    if no_crop:
        aligned_img = align_image_no_crop(template_img, best_M, reference_img.shape)
    else:
        aligned_img = align_image(template_img, best_M, reference_img.shape)
    
    if aligned_img is None:
        print("Error: Failed to apply transformation")
        return False
    
    # Visualize matches
    print("Visualizing matches...")
    img_matches = visualize_matches(reference_img, template_img, best_info)
    
    if img_matches is not None:
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches - {best_method.upper()} ({best_info["inliers"]} inliers)', 
                  fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Visualize alignment
    print("Visualizing alignment...")
    visualize_alignment(reference_img, template_img, aligned_img)
    
    # Save aligned image
    success = cv2.imwrite(output_path, aligned_img)
    if success:
        print(f"\n✓ Aligned image saved to: {output_path}")
        print(f"  Output size: {aligned_img.shape}")
    else:
        print(f"\n✗ Failed to save aligned image to: {output_path}")
    
    # Print homography matrix
    print("\nHomography Matrix:")
    print(best_M)
    
    print("\n" + "="*70)
    print("TIPS FOR NATURAL SCENES (Mountains, Vegetation, Terrain):")
    print("="*70)
    print("1. SIFT usually performs BEST for natural outdoor scenes")
    print("2. If matching fails, try these adjustments:")
    print("   - Use --method auto to try all methods")
    print("   - Use --use-mask to focus on rocky/stable regions")
    print("   - Ensure good lighting and image overlap")
    print("3. Use --no-crop to preserve full transformed image")
    print("4. Use --crop to match reference image dimensions")
    print("="*70)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Align images for natural scene monitoring (landslides, terrain changes)'
    )
    parser.add_argument('--reference', '-r', type=str, default="cropped_from_image_1.JPG",
                        help='Path to reference image')
    parser.add_argument('--template', '-t', type=str, 
                        default="Foundation Dataset/Location 1/142958_0831_2025.JPG",
                        help='Path to image to be aligned')
    parser.add_argument('--output', '-o', type=str, default="aligned_image.jpg",
                        help='Path to save aligned image')
    parser.add_argument('--method', '-m', type=str, default='auto',
                        choices=['sift', 'orb', 'akaze', 'auto'],
                        help='Feature detection method (auto tries all)')
    parser.add_argument('--use-mask', action='store_true',
                        help='Focus on stable regions (rocks, not vegetation)')
    parser.add_argument('--crop', action='store_true',
                        help='Crop output to reference image size (default: no crop)')
    
    args = parser.parse_args()
    
    # Run main pipeline
    success = main(
        reference_path=args.reference,
        template_path=args.template,
        output_path=args.output,
        method=args.method,
        use_mask=args.use_mask,
        no_crop=not args.crop  # By default, no_crop=True
    )
    
    sys.exit(0 if success else 1)
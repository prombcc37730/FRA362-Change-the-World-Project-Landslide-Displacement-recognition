"""
Multi-ROI Template Matching - Try Multiple Markers and Use the Best One

This approach:
1. Tries tracking each ROI independently
2. Ranks them by confidence score
3. Uses the best matching ROI for transformation
4. Falls back to next best if feature extraction fails

Use case: When you have multiple potential markers but don't know which is most reliable
"""

import cv2
import numpy as np
import math
import sys
from typing import Tuple, Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def track_single_marker(
    imgA: np.ndarray,
    imgB: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    expansion_factor: float,
    scales: list,
    confidence_threshold: float,
    marker_id: int
) -> Optional[Dict]:
    """
    Track a single marker between two images using multi-scale template matching.
    
    Returns:
        dict with tracking results or None if tracking failed
    """
    
    roiA_x, roiA_y, roiA_w, roiA_h = roi_coords
    template = imgA[roiA_y:roiA_y+roiA_h, roiA_x:roiA_x+roiA_w]
    
    if template.size == 0:
        logger.error(f"Marker {marker_id}: Invalid ROI coordinates: {roi_coords}")
        return None

    roiA_center = np.array([roiA_x + roiA_w/2, roiA_y + roiA_h/2])

    # Define search region (centered expansion)
    expanded_w = int(roiA_w * expansion_factor)
    expanded_h = int(roiA_h * expansion_factor)

    search_x = int(roiA_center[0] - expanded_w / 2)
    search_y = int(roiA_center[1] - expanded_h / 2)

    # Clamp to image boundaries
    search_x = max(0, search_x)
    search_y = max(0, search_y)
    search_w = min(imgB.shape[1] - search_x, expanded_w)
    search_h = min(imgB.shape[0] - search_y, expanded_h)

    if search_x + search_w > imgB.shape[1]:
        search_w = imgB.shape[1] - search_x
    if search_y + search_h > imgB.shape[0]:
        search_h = imgB.shape[0] - search_y

    search_region = imgB[search_y:search_y+search_h, search_x:search_x+search_w]

    # Validate search region
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        logger.error(f"Marker {marker_id}: Search region too small")
        return None

    # Multi-scale template matching
    best_match = None
    best_val = -1
    best_scale = 1.0

    for scale in scales:
        if scale != 1.0:
            scaled_w = int(template.shape[1] * scale)
            scaled_h = int(template.shape[0] * scale)
            if scaled_w < 1 or scaled_h < 1:
                continue
            template_scaled = cv2.resize(template, (scaled_w, scaled_h))
        else:
            template_scaled = template.copy()
        
        if template_scaled.shape[0] > search_region.shape[0] or \
           template_scaled.shape[1] > search_region.shape[1]:
            continue
        
        result = cv2.matchTemplate(search_region, template_scaled, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc
            best_scale = scale

    # Validate match
    if best_match is None or best_val < confidence_threshold:
        return None

    # Calculate matched center
    matched_x = search_x + best_match[0] + int(roiA_w * best_scale / 2)
    matched_y = search_y + best_match[1] + int(roiA_h * best_scale / 2)
    matched_center = np.array([matched_x, matched_y])

    displacement = matched_center - roiA_center
    distance = np.linalg.norm(displacement)

    return {
        'marker_id': marker_id,
        'roi_coords': roi_coords,
        'original_center': roiA_center,
        'matched_center': matched_center,
        'displacement': displacement,
        'distance': distance,
        'scale': best_scale,
        'confidence': best_val,
        'search_region': (search_x, search_y, search_w, search_h),
        'template': template,
        'search_region_img': search_region
    }


def extract_features_from_marker(
    imgA: np.ndarray,
    imgB: np.ndarray,
    marker_result: Dict,
    lowe_ratio: float
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract and match SIFT features for a single marker.
    
    Returns:
        src_pts, dst_pts, num_matches: Feature correspondences and count
    """
    
    roiA_x, roiA_y, roiA_w, roiA_h = marker_result['roi_coords']
    search_x, search_y, search_w, search_h = marker_result['search_region']
    
    # Extract ROI from Image A
    roiA_region = imgA[roiA_y:roiA_y+roiA_h, roiA_x:roiA_x+roiA_w]
    search_region = imgB[search_y:search_y+search_h, search_x:search_x+search_w]
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect features
    kp_roiA, des_roiA = sift.detectAndCompute(roiA_region, None)
    kp_search, des_search = sift.detectAndCompute(search_region, None)
    
    if len(kp_roiA) < 4 or len(kp_search) < 4:
        return np.array([]), np.array([]), 0
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_roiA, des_search, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return np.array([]), np.array([]), 0
    
    # Extract matched keypoints and convert to full image coordinates
    src_pts_roi = np.float32([kp_roiA[m.queryIdx].pt for m in good_matches])
    dst_pts_search = np.float32([kp_search[m.trainIdx].pt for m in good_matches])
    
    # Convert to full image coordinates
    src_pts = src_pts_roi + np.array([roiA_x, roiA_y])
    dst_pts = dst_pts_search + np.array([search_x, search_y])
    
    return src_pts, dst_pts, len(good_matches)


def process_images_best_roi(
    img_a_path: str,
    img_b_path: str,
    roi_coords_list: List[Tuple[int, int, int, int]],
    expansion_factor: float = 3.0,
    scales: list = None,
    confidence_threshold: float = 0.3,
    lowe_ratio: float = 0.75,
    min_feature_matches: int = 10
):
    """
    Try multiple ROIs and use the best one for transformation.
    
    Strategy:
    1. Track all ROIs using template matching
    2. Rank by confidence score
    3. Try feature extraction on best candidates
    4. Use the best one that has sufficient features
    
    Args:
        img_a_path: Path to first image (reference)
        img_b_path: Path to second image (to track)
        roi_coords_list: List of (x, y, width, height) tuples - tries each one
        expansion_factor: Search region expansion factor
        scales: List of scale factors for template matching
        confidence_threshold: Minimum template match confidence
        lowe_ratio: Lowe's ratio test threshold for feature matching
        min_feature_matches: Minimum features needed for homography
    
    Returns:
        dict: Results including best ROI, transformation matrix, and rankings
    """
    
    # Validate parameters
    if expansion_factor < 1.0:
        raise ValueError("expansion_factor must be >= 1.0")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Load images
    logger.info("Loading images...")
    imgA = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)

    if imgA is None or imgB is None:
        raise ValueError("Error: Could not load images. Check file paths.")

    logger.info(f"Image A: {imgA.shape}")
    logger.info(f"Image B: {imgB.shape}")
    logger.info(f"Testing {len(roi_coords_list)} candidate ROIs")

    print(f"\n{'='*70}")
    print(f" TESTING MULTIPLE ROI CANDIDATES")
    print(f"{'='*70}")
    print(f"Total candidates: {len(roi_coords_list)}")
    print(f"Strategy: Try all, rank by confidence, use best one")

    # Track all ROI candidates
    print(f"\n{'='*70}")
    print(f" PHASE 1: Template Matching All Candidates")
    print(f"{'='*70}")
    
    candidates = []
    for i, roi_coords in enumerate(roi_coords_list):
        print(f"\n--- Candidate {i+1}/{len(roi_coords_list)} ---")
        print(f"ROI: ({roi_coords[0]}, {roi_coords[1]}) size {roi_coords[2]}x{roi_coords[3]}")
        
        result = track_single_marker(
            imgA, imgB, roi_coords, expansion_factor, 
            scales, confidence_threshold, marker_id=i+1
        )
        
        if result is not None:
            candidates.append(result)
            print(f"✓ Match found! Confidence: {result['confidence']:.3f}")
            print(f"  Scale: {result['scale']:.2f}x")
            print(f"  Displacement: {result['distance']:.1f}px")
        else:
            print(f"✗ No match (confidence < {confidence_threshold})")

    if len(candidates) == 0:
        logger.error("No ROI candidates successfully tracked!")
        sys.exit(1)

    # Rank candidates by confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)

    print(f"\n{'='*70}")
    print(f" RANKING RESULTS")
    print(f"{'='*70}")
    print(f"Successfully tracked: {len(candidates)}/{len(roi_coords_list)} candidates")
    print(f"\nRanked by confidence:")
    for i, cand in enumerate(candidates):
        print(f"  {i+1}. Marker {cand['marker_id']}: confidence={cand['confidence']:.3f}, "
              f"scale={cand['scale']:.2f}x, displacement={cand['distance']:.1f}px")

    # Try feature extraction on candidates in order
    print(f"\n{'='*70}")
    print(f" PHASE 2: Feature Extraction (Testing Best Candidates)")
    print(f"{'='*70}")

    best_marker = None
    best_features = None
    best_homography = None
    
    for i, candidate in enumerate(candidates):
        marker_id = candidate['marker_id']
        print(f"\nTrying candidate #{i+1} (Marker {marker_id}, confidence={candidate['confidence']:.3f})...")
        
        src_pts, dst_pts, num_matches = extract_features_from_marker(
            imgA, imgB, candidate, lowe_ratio
        )
        
        if num_matches < min_feature_matches:
            print(f"  ✗ Not enough features ({num_matches} < {min_feature_matches})")
            continue
        
        print(f"  ✓ Found {num_matches} feature matches")
        
        # Try to compute homography
        try:
            H, mask = cv2.findHomography(
                src_pts.reshape(-1, 1, 2),
                dst_pts.reshape(-1, 1, 2),
                cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if H is None:
                print(f"  ✗ Homography computation failed")
                continue
            
            num_inliers = np.sum(mask)
            inlier_ratio = num_inliers / num_matches
            
            print(f"  ✓ Homography computed: {num_inliers}/{num_matches} inliers ({inlier_ratio*100:.1f}%)")
            
            # Check if this is good enough
            if inlier_ratio > 0.5 and num_inliers >= 10:
                print(f"  ✓✓ SELECTED as best candidate!")
                best_marker = candidate
                best_features = (src_pts, dst_pts, num_matches, num_inliers)
                best_homography = H
                break
            else:
                print(f"  ⚠ Quality not sufficient (inliers: {num_inliers}, ratio: {inlier_ratio:.2f})")
                
        except Exception as e:
            print(f"  ✗ Error computing homography: {e}")
            continue

    # Decide on final transformation
    if best_marker is None:
        logger.warning("No candidate passed feature extraction. Using best template match only.")
        best_marker = candidates[0]  # Use highest confidence
        
        # Simple translation transformation
        tx = best_marker['displacement'][0]
        ty = best_marker['displacement'][1]
        
        H_A_to_B = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        homography_available = False
        num_inliers = 0
        num_matches = 0
    else:
        H_A_to_B = best_homography
        homography_available = True
        src_pts, dst_pts, num_matches, num_inliers = best_features

    print(f"\n{'='*70}")
    print(f" FINAL SELECTION")
    print(f"{'='*70}")
    print(f"Selected: Marker {best_marker['marker_id']}")
    print(f"  ROI: ({best_marker['roi_coords'][0]}, {best_marker['roi_coords'][1]}) "
          f"size {best_marker['roi_coords'][2]}x{best_marker['roi_coords'][3]}")
    print(f"  Template confidence: {best_marker['confidence']:.3f}")
    print(f"  Template scale: {best_marker['scale']:.2f}x")
    print(f"  Displacement: {best_marker['distance']:.1f}px")
    
    if homography_available:
        print(f"  Feature matches: {num_inliers}/{num_matches} inliers")
        print(f"  Transformation: HOMOGRAPHY (8 DOF)")
    else:
        print(f"  Transformation: TRANSLATION ONLY (2 DOF)")

    # Analyze transformation
    print(f"\n{'='*70}")
    print(f" TRANSFORMATION MATRIX (Image A → Image B)")
    print(f"{'='*70}")
    print(H_A_to_B)

    if homography_available:
        # Extract transformation properties
        scale_x = np.linalg.norm(H_A_to_B[0, :2])
        scale_y = np.linalg.norm(H_A_to_B[1, :2])
        avg_scale = (scale_x + scale_y) / 2
        rotation_rad = math.atan2(H_A_to_B[1, 0], H_A_to_B[0, 0])
        rotation_deg = math.degrees(rotation_rad)

        print(f"\nTransformation properties:")
        print(f"  Scale X: {scale_x:.4f}x ({(scale_x-1)*100:+.2f}%)")
        print(f"  Scale Y: {scale_y:.4f}x ({(scale_y-1)*100:+.2f}%)")
        print(f"  Avg Scale: {avg_scale:.4f}x ({(avg_scale-1)*100:+.2f}%)")
        print(f"  Rotation: {rotation_deg:+.2f}°")
    else:
        avg_scale = 1.0
        rotation_deg = 0.0
        print(f"\nSimple translation only (feature extraction failed)")

    # Compute inverse
    try:
        H_B_to_A = np.linalg.inv(H_A_to_B)
    except np.linalg.LinAlgError:
        logger.error("Failed to compute inverse transformation")
        sys.exit(1)

    # Align Image B to Image A
    print(f"\n{'='*70}")
    print(f" ALIGNING IMAGES")
    print(f"{'='*70}")

    imgB_aligned = cv2.warpPerspective(imgB, H_B_to_A, (imgA.shape[1], imgA.shape[0]))
    cv2.imwrite("imgB_aligned_to_imgA.jpg", imgB_aligned)
    logger.info("✓ Saved: imgB_aligned_to_imgA.jpg")

    # Create comparison images
    diff = cv2.absdiff(imgA, imgB_aligned)
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imwrite("alignment_difference.jpg", diff_colored)

    comparison = np.hstack([imgA, imgB_aligned])
    cv2.imwrite("comparison_A_vs_B_aligned.jpg", comparison)

    overlay = cv2.addWeighted(imgA, 0.5, imgB_aligned, 0.5, 0)
    cv2.imwrite("overlay_blend.jpg", overlay)

    # Visualizations
    print(f"\n{'='*70}")
    print(f" CREATING VISUALIZATIONS")
    print(f"{'='*70}")

    # Image A with all tested ROIs (winners and losers)
    imgA_vis = cv2.cvtColor(imgA.copy(), cv2.COLOR_GRAY2BGR)
    
    for i, cand in enumerate(candidates):
        # Best marker = green, others = gray
        if cand['marker_id'] == best_marker['marker_id']:
            color = (0, 255, 0)  # Green (winner)
            thickness = 4
            label = f"SELECTED M{cand['marker_id']}"
        else:
            color = (128, 128, 128)  # Gray (not selected)
            thickness = 2
            label = f"M{cand['marker_id']}"
        
        roiA_x, roiA_y, roiA_w, roiA_h = cand['roi_coords']
        center = cand['original_center']
        
        cv2.rectangle(imgA_vis, (roiA_x, roiA_y), 
                     (roiA_x+roiA_w, roiA_y+roiA_h), color, thickness)
        cv2.circle(imgA_vis, (int(center[0]), int(center[1])), 10, color, -1)
        cv2.putText(imgA_vis, label, (roiA_x-10, roiA_y-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imwrite("imageA_with_roi_candidates.jpg", imgA_vis)

    # Image B with best matched marker
    imgB_vis = cv2.cvtColor(imgB.copy(), cv2.COLOR_GRAY2BGR)
    
    orig_center = best_marker['original_center']
    matched_center = best_marker['matched_center']
    displacement = best_marker['displacement']
    distance = best_marker['distance']
    
    # Draw matched position
    cv2.circle(imgB_vis, (int(matched_center[0]), int(matched_center[1])), 10, (0, 255, 0), -1)
    
    # Draw arrow from original to matched
    cv2.arrowedLine(imgB_vis,
                   (int(orig_center[0]), int(orig_center[1])),
                   (int(matched_center[0]), int(matched_center[1])),
                   (0, 255, 0), 4, tipLength=0.03)
    
    # Add label
    label = f"Best: M{best_marker['marker_id']}, {distance:.0f}px, conf={best_marker['confidence']:.2f}"
    cv2.putText(imgB_vis, label,
               (int(matched_center[0])-200, int(matched_center[1])-30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imwrite("imageB_with_best_match.jpg", imgB_vis)

    # Feature matches visualization (if available)
    if homography_available:
        roiA_x, roiA_y, roiA_w, roiA_h = best_marker['roi_coords']
        search_x, search_y, search_w, search_h = best_marker['search_region']
        
        roiA_region = imgA[roiA_y:roiA_y+roiA_h, roiA_x:roiA_x+roiA_w]
        search_region = imgB[search_y:search_y+search_h, search_x:search_x+search_w]
        
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(roiA_region, None)
        kp2, des2 = sift.detectAndCompute(search_region, None)
        
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < lowe_ratio * n.distance:
                    good.append(m)
        
        match_vis = cv2.drawMatches(
            roiA_region, kp1, search_region, kp2, good[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("feature_matches_best_roi.jpg", match_vis)
        logger.info("✓ Saved: feature_matches_best_roi.jpg")

    print(f"\n{'='*70}")
    print(f" OUTPUT FILES")
    print(f"{'='*70}")
    print("✓ imgB_aligned_to_imgA.jpg - Aligned Image B")
    print("✓ alignment_difference.jpg - Difference heatmap")
    print("✓ comparison_A_vs_B_aligned.jpg - Side by side")
    print("✓ overlay_blend.jpg - 50/50 overlay")
    print("✓ imageA_with_roi_candidates.jpg - All tested ROIs (winner in green)")
    print("✓ imageB_with_best_match.jpg - Best match + displacement")
    if homography_available:
        print("✓ feature_matches_best_roi.jpg - Feature correspondences")
    print(f"{'='*70}")

    # Summary
    print(f"\n{'='*70}")
    print(f" SUMMARY")
    print(f"{'='*70}")
    print(f"Tested ROIs: {len(roi_coords_list)}")
    print(f"Successfully matched: {len(candidates)}")
    print(f"Selected best: Marker {best_marker['marker_id']}")
    print(f"\nBest marker details:")
    print(f"  Template confidence: {best_marker['confidence']:.3f} (rank #{1})")
    print(f"  Displacement: {best_marker['distance']:.1f}px")
    print(f"  Direction: {math.degrees(math.atan2(best_marker['displacement'][1], best_marker['displacement'][0])):.1f}°")
    
    if homography_available:
        print(f"  Features: {num_inliers} inliers from {num_matches} matches")
        print(f"  Global scale: {avg_scale:.4f}x ({(avg_scale-1)*100:+.2f}%)")
        print(f"  Rotation: {rotation_deg:+.2f}°")
    
    print(f"\nAll candidates ranked:")
    for i, cand in enumerate(candidates):
        status = "✓ SELECTED" if cand['marker_id'] == best_marker['marker_id'] else ""
        print(f"  {i+1}. Marker {cand['marker_id']}: conf={cand['confidence']:.3f}, "
              f"disp={cand['distance']:.1f}px {status}")

    print(f"{'='*70}")

    # Return results
    return {
        'transformation_matrix': H_A_to_B,
        'inverse_transformation_matrix': H_B_to_A,
        'best_marker': best_marker,
        'all_candidates': candidates,
        'num_candidates_tested': len(roi_coords_list),
        'num_candidates_matched': len(candidates),
        'homography_available': homography_available,
        'num_feature_matches': num_inliers if homography_available else 0,
        'scale': avg_scale if homography_available else 1.0,
        'rotation_degrees': rotation_deg if homography_available else 0.0
    }


def main():
    """Main execution - tries multiple ROIs and picks best one."""
    
    # Configuration
    img_a_path = "Foundation Dataset/Location 1/083142_0831_2025.JPG"
    img_b_path = "Foundation Dataset/Location 1/134123_0830_2025.JPG"
    
    # Define multiple ROI candidates (x, y, width, height)
    # The algorithm will try each one and pick the best
    roi_coords_list = [
        (513, 2360, 243, 153),    # Candidate 1
        (2480, 2540, 176, 133),   # Candidate 2
        # (2400, 2320, 280, 236),   # Candidate 3
        # (496, 1200, 280, 236),    # Candidate 4
        # (1800, 800, 280, 236),    # Candidate 5
    ]
    
    # Processing parameters
    expansion_factor = 3.0
    confidence_threshold = 0.3
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    lowe_ratio = 0.75
    min_feature_matches = 10
    
    try:
        results = process_images_best_roi(
            img_a_path=img_a_path,
            img_b_path=img_b_path,
            roi_coords_list=roi_coords_list,
            expansion_factor=expansion_factor,
            scales=scales,
            confidence_threshold=confidence_threshold,
            lowe_ratio=lowe_ratio,
            min_feature_matches=min_feature_matches
        )
        
        logger.info("Processing completed successfully!")
        print(f"\n🎯 Best marker: #{results['best_marker']['marker_id']}")
        return results
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
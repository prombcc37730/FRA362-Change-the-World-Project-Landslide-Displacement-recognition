import cv2
import numpy as np
import math

# ---------------------------------------
# Load images
# ---------------------------------------
imgA = cv2.imread("Foundation Dataset/Location 1/083142_0831_2025.JPG", cv2.IMREAD_GRAYSCALE)
imgB = cv2.imread("Foundation Dataset/Location 1/154809_0831_2025.JPG", cv2.IMREAD_GRAYSCALE)

if imgA is None or imgB is None:
    raise ValueError("Error: Could not load images")

print(f"Image A: {imgA.shape}")
print(f"Image B: {imgB.shape}")

# ---------------------------------------
# Define ROI (marker/target to track)
# ---------------------------------------
roiA_x, roiA_y, roiA_w, roiA_h = 496, 2320, 280, 236
template = imgA[roiA_y:roiA_y+roiA_h, roiA_x:roiA_x+roiA_w]

roiA_center = np.array([roiA_x + roiA_w/2, roiA_y + roiA_h/2])

print(f"\n{'='*60}")
print(f" Template (ROI) Configuration")
print(f"{'='*60}")
print(f"Position in Image A: ({roiA_x}, {roiA_y})")
print(f"Size: {roiA_w}x{roiA_h}")
print(f"Center: ({roiA_center[0]:.1f}, {roiA_center[1]:.1f})")

# ---------------------------------------
# Define search region (centered, 1.5x expansion)
# Your smart approach: expand from ROI center
# ---------------------------------------
# Adjustable expansion factor based on expected movement
# 1.5x = ±70px for 280x236 ROI (gradual landslide)
# 2.0x = ±140px (moderate movement)
# 3.0x = ±280px (large movement)
expansion_factor = 3  # Start with your suggestion

# Calculate expanded dimensions
expanded_w = int(roiA_w * expansion_factor)
expanded_h = int(roiA_h * expansion_factor)

# Center the search region on ROI center
search_x = int(roiA_center[0] - expanded_w / 2)
search_y = int(roiA_center[1] - expanded_h / 2)

# Clamp to image boundaries
search_x = max(0, search_x)
search_y = max(0, search_y)
search_w = min(imgB.shape[1] - search_x, expanded_w)
search_h = min(imgB.shape[0] - search_y, expanded_h)

# Adjust if search region extends beyond image
if search_x + search_w > imgB.shape[1]:
    search_w = imgB.shape[1] - search_x
if search_y + search_h > imgB.shape[0]:
    search_h = imgB.shape[0] - search_y

search_region = imgB[search_y:search_y+search_h, search_x:search_x+search_w]

# Validate search region
if search_w < roiA_w * 1.2 or search_h < roiA_h * 1.2:
    print(f"\n⚠ WARNING: Search region too small!")
    print(f"  Search: {search_w}x{search_h}")
    print(f"  Template: {roiA_w}x{roiA_h}")
    print(f"  This may fail if object moved significantly.")
    print(f"  Consider increasing expansion_factor to 2.0 or higher")

if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
    raise ValueError(f"Search region ({search_w}x{search_h}) smaller than template ({roiA_w}x{roiA_h}). Increase expansion_factor.")

print(f"\n{'='*60}")
print(f" Search Region Configuration")
print(f"{'='*60}")
print(f"Expansion factor: {expansion_factor}x")
print(f"Original ROI: {roiA_w}x{roiA_h}")
print(f"Search region: {search_w}x{search_h}")
print(f"Search position: ({search_x}, {search_y})")
print(f"Max displacement allowed: ±{int((expanded_w - roiA_w) / 2)} pixels")

# Save debug images
cv2.imwrite("debug_template.jpg", template)
cv2.imwrite("debug_search_region.jpg", search_region)

# Visualize search strategy
imgB_search_vis = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)
# Original ROI position (green)
cv2.rectangle(imgB_search_vis, (roiA_x, roiA_y), (roiA_x+roiA_w, roiA_y+roiA_h), (0, 255, 0), 3)
cv2.putText(imgB_search_vis, "Expected position", (roiA_x, roiA_y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# Search region (blue)
cv2.rectangle(imgB_search_vis, (search_x, search_y), (search_x+search_w, search_y+search_h), (255, 0, 0), 3)
cv2.putText(imgB_search_vis, f"Search region ({expansion_factor}x)", 
            (search_x, search_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
# Center point
cv2.circle(imgB_search_vis, (int(roiA_center[0]), int(roiA_center[1])), 8, (255, 255, 0), -1)
cv2.imwrite("search_strategy_visualization.jpg", imgB_search_vis)
print("✓ Saved: search_strategy_visualization.jpg")

# ---------------------------------------
# Multi-scale Template Matching
# (handles scale changes from perspective/distance)
# ---------------------------------------
scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # Try different scales
best_match = None
best_val = -1
best_scale = 1.0

print(f"\n{'='*60}")
print(f" Multi-Scale Template Matching")
print(f"{'='*60}")

for scale in scales:
    # Resize template
    if scale != 1.0:
        scaled_w = int(template.shape[1] * scale)
        scaled_h = int(template.shape[0] * scale)
        template_scaled = cv2.resize(template, (scaled_w, scaled_h))
    else:
        template_scaled = template
    
    # Skip if template bigger than search region
    if template_scaled.shape[0] > search_region.shape[0] or \
       template_scaled.shape[1] > search_region.shape[1]:
        continue
    
    # Match using normalized cross-correlation
    result = cv2.matchTemplate(search_region, template_scaled, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    print(f"Scale {scale:.2f}: confidence={max_val:.3f}, location=({max_loc[0]}, {max_loc[1]})")
    
    if max_val > best_val:
        best_val = max_val
        best_match = max_loc
        best_scale = scale

# ---------------------------------------
# Calculate matched position in full Image B
# ---------------------------------------
if best_match is None or best_val < 0.3:
    print(f"\n{'='*60}")
    print(f" ✗ MATCHING FAILED")
    print(f"{'='*60}")
    print(f"Best confidence: {best_val:.3f} (threshold: 0.3)")
    print("\nPossible reasons:")
    print("  • Marker not visible in Image B")
    print("  • Marker damaged or occluded")
    print("  • Too much change between images (weather, lighting)")
    print("  • Search region doesn't contain marker")
    print("\nSuggestions:")
    print("  • Increase search_margin (currently {search_margin}px)")
    print("  • Check if marker visible in both images")
    print("  • Reduce time between image captures")
    exit(1)

matched_x = search_x + best_match[0] + int(roiA_w * best_scale / 2)
matched_y = search_y + best_match[1] + int(roiA_h * best_scale / 2)
matched_center = np.array([matched_x, matched_y])

displacement = matched_center - roiA_center
distance = np.linalg.norm(displacement)

print(f"\n{'='*60}")
print(f" ✓ MATCH FOUND!")
print(f"{'='*60}")
print(f"Best scale: {best_scale:.2f}x")
print(f"Confidence: {best_val:.3f} ({best_val*100:.1f}%)")
print(f"\nOriginal center (A): ({roiA_center[0]:.1f}, {roiA_center[1]:.1f})")
print(f"Matched center (B):  ({matched_center[0]:.1f}, {matched_center[1]:.1f})")
print(f"\nDisplacement:")
print(f"  ΔX = {displacement[0]:+.1f} pixels")
print(f"  ΔY = {displacement[1]:+.1f} pixels")
print(f"  Distance = {distance:.1f} pixels")
print(f"  Direction = {math.degrees(math.atan2(displacement[1], displacement[0])):.1f}°")

# Scale change
scale_change = (best_scale - 1.0) * 100
print(f"\nScale change: {scale_change:+.1f}%")

# ---------------------------------------
# Compute transformation matrix
# ---------------------------------------
# For template matching, we have translation + scale
# Create affine transformation matrix
tx = displacement[0]
ty = displacement[1]
s = best_scale

# Affine matrix: scale + translation
A_2x3 = np.array([
    [s, 0, tx],
    [0, s, ty]
], dtype=np.float32)

# Convert to 3x3
A_3x3 = np.vstack([A_2x3, [0, 0, 1]])

print(f"\n{'='*60}")
print(f" TRANSFORMATION MATRIX (3x3)")
print(f"{'='*60}")
print(A_3x3)
print("\nThis transforms Image B to align with Image A")

# ---------------------------------------
# Apply inverse transformation to align Image B
# ---------------------------------------
# Inverse: scale back and translate back
H_B_to_A = np.array([
    [1/s, 0, -tx/s],
    [0, 1/s, -ty/s],
    [0, 0, 1]
], dtype=np.float32)

print(f"\n{'='*60}")
print(f" INVERSE TRANSFORMATION (3x3)")
print(f"{'='*60}")
print(H_B_to_A)

# Warp Image B to align with Image A
imgB_aligned = cv2.warpPerspective(imgB, H_B_to_A, (imgA.shape[1], imgA.shape[0]))
cv2.imwrite("imgB_aligned_to_imgA.jpg", imgB_aligned)

# ---------------------------------------
# Create difference/comparison images
# ---------------------------------------
# Difference map
diff = cv2.absdiff(imgA, imgB_aligned)
diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
cv2.imwrite("alignment_difference.jpg", diff_colored)

# Side-by-side comparison
comparison = np.hstack([imgA, imgB_aligned])
cv2.imwrite("comparison_A_vs_B_aligned.jpg", comparison)

# Overlay (50/50 blend)
overlay = cv2.addWeighted(imgA, 0.5, imgB_aligned, 0.5, 0)
cv2.imwrite("overlay_blend.jpg", overlay)

# ---------------------------------------
# Visualizations
# ---------------------------------------
# Image A with ROI
imgA_vis = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
cv2.rectangle(imgA_vis, (roiA_x, roiA_y), (roiA_x+roiA_w, roiA_y+roiA_h), (0, 255, 0), 4)
cv2.circle(imgA_vis, (int(roiA_center[0]), int(roiA_center[1])), 10, (0, 255, 0), -1)
cv2.putText(imgA_vis, "Template", (roiA_x-10, roiA_y-15), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
cv2.imwrite("imageA_with_template.jpg", imgA_vis)

# Image B with matched location
imgB_vis = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)
matched_w = int(roiA_w * best_scale)
matched_h = int(roiA_h * best_scale)
matched_x1 = search_x + best_match[0]
matched_y1 = search_y + best_match[1]

cv2.rectangle(imgB_vis, 
              (matched_x1, matched_y1), 
              (matched_x1 + matched_w, matched_y1 + matched_h), 
              (0, 0, 255), 4)
cv2.circle(imgB_vis, (int(matched_center[0]), int(matched_center[1])), 10, (0, 0, 255), -1)

# Draw arrow showing displacement
cv2.arrowedLine(imgB_vis,
                (int(roiA_center[0]), int(roiA_center[1])),
                (int(matched_center[0]), int(matched_center[1])),
                (255, 0, 255), 4, tipLength=0.03)

# Add text showing displacement
text = f"Moved: {distance:.0f}px"
cv2.putText(imgB_vis, text, 
            (int(matched_center[0])-100, int(matched_center[1])-30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

cv2.imwrite("imageB_with_match.jpg", imgB_vis)

# Heat map of match confidence
result_vis = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
result_vis = cv2.normalize(result_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
result_colored = cv2.applyColorMap(result_vis, cv2.COLORMAP_JET)
cv2.imwrite("match_confidence_heatmap.jpg", result_colored)

print(f"\n{'='*60}")
print(f" OUTPUT FILES")
print(f"{'='*60}")
print("✓ imgB_aligned_to_imgA.jpg - Aligned Image B")
print("✓ alignment_difference.jpg - Difference heatmap")
print("✓ comparison_A_vs_B_aligned.jpg - Side by side")
print("✓ overlay_blend.jpg - 50/50 overlay")
print("✓ imageA_with_template.jpg - Template location")
print("✓ imageB_with_match.jpg - Match location + arrow")
print("✓ match_confidence_heatmap.jpg - Confidence map")
print(f"{'='*60}")

print(f"\n{'='*60}")
print(f" MONITORING SUMMARY")
print(f"{'='*60}")
print(f"Search strategy: Centered expansion ({expansion_factor}x)")
print(f"Search efficiency: {((roiA_w*roiA_h*expansion_factor**2)/(imgB.shape[0]*imgB.shape[1])*100):.2f}% of full image")
print(f"\nMarker displacement: {distance:.1f} pixels")
print(f"  ΔX = {displacement[0]:+.1f} pixels")
print(f"  ΔY = {displacement[1]:+.1f} pixels")
print(f"Direction: {math.degrees(math.atan2(displacement[1], displacement[0])):.1f}°")
print(f"Scale change: {scale_change:+.1f}%")
print(f"Match confidence: {best_val*100:.1f}%")

# Check if displacement is within expected range
max_displacement = int((expanded_w - roiA_w) / 2)
if distance > max_displacement * 0.8:
    print(f"\n⚠ WARNING: Displacement ({distance:.0f}px) near search limit ({max_displacement}px)")
    print(f"  Consider increasing expansion_factor to {expansion_factor * 1.5:.1f}x")
elif distance < max_displacement * 0.2:
    print(f"\n✓ Small displacement - could reduce expansion_factor to {expansion_factor * 0.8:.1f}x for speed")

print(f"{'='*60}")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# The Constants from the C++ code (base tree shape)
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

# Simulate a Poly struct: transformed tree at position (1.5, 1.0) with slight rotation
# This represents what px[NV] and py[NV] would contain after transformation
import math
angle = 15 * math.pi / 180  # 15 degrees rotation
cx, cy = 1.5, 1.0  # Translation (center position)

px = []
py = []
for i in range(len(TX)):
    # Rotate and translate (simulating the transformation)
    x_rot = TX[i] * math.cos(angle) - TY[i] * math.sin(angle)
    y_rot = TX[i] * math.sin(angle) + TY[i] * math.cos(angle)
    px.append(x_rot + cx)
    py.append(y_rot + cy)

# Calculate bounding box (what x0, y0, x1, y1 would be)
x0 = min(px)
y0 = min(py)
x1 = max(px)
y1 = max(py)

# Close the loop for plotting
px_plot = px + [px[0]]
py_plot = py + [py[0]]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# ===== LEFT PLOT: Original Tree =====
TX_plot = TX + [TX[0]]
TY_plot = TY + [TY[0]]
ax1.plot(TX_plot, TY_plot, 'g-', linewidth=2, label='Tree Polygon')
ax1.fill(TX_plot, TY_plot, 'g', alpha=0.3)
ax1.scatter(TX, TY, color='red', s=50, zorder=5)
ax1.set_title("Original Tree Shape (TX, TY)\nBase coordinates before transformation")
ax1.axis('equal')
ax1.grid(True, alpha=0.3)
ax1.legend()

# ===== RIGHT PLOT: Poly Struct Visualization =====
# Draw bounding box (x0, y0, x1, y1)
bbox_width = x1 - x0
bbox_height = y1 - y0
bbox_rect = patches.Rectangle((x0, y0), bbox_width, bbox_height, 
                               linewidth=2, edgecolor='blue', 
                               facecolor='none', linestyle='--', 
                               label='Bounding Box (x0,y0,x1,y1)')
ax2.add_patch(bbox_rect)

# Draw the polygon (px, py arrays)
ax2.plot(px_plot, py_plot, 'g-', linewidth=2, label='Polygon Vertices (px, py)')
ax2.fill(px_plot, py_plot, 'g', alpha=0.3)
ax2.scatter(px, py, color='red', s=50, zorder=5, label='15 Vertices')

# Mark bounding box corners
ax2.scatter([x0, x1, x0, x1], [y0, y0, y1, y1], color='blue', s=100, 
            marker='s', zorder=6, label='Bounding Box Corners')
ax2.annotate(f'(x0={x0:.2f}, y0={y0:.2f})', (x0, y0), 
             xytext=(10, -20), textcoords='offset points', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
ax2.annotate(f'(x1={x1:.2f}, y1={y1:.2f})', (x1, y1), 
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax2.set_title("Poly Struct Visualization\npx[NV], py[NV] = vertices\nx0,y0,x1,y1 = bounding box")
ax2.axis('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('tree_visualization.png', dpi=150, bbox_inches='tight')
print("Tree visualization saved as tree_visualization.png")
print(f"\nPoly struct contains:")
print(f"  - px[NV], py[NV]: {len(px)} vertices (the tree shape)")
print(f"  - Bounding box: x0={x0:.3f}, y0={y0:.3f}, x1={x1:.3f}, y1={y1:.3f}")

# ============================================================================
# RAY CASTING VISUALIZATION (Point-In-Polygon Test)
# ============================================================================

# Implement the pip function from C++ code
def pip(px_test, py_test, q_px, q_py):
    """Point-in-polygon test using ray casting algorithm"""
    in_polygon = False
    j = len(q_px) - 1  # Last vertex
    
    crossings = []  # Store crossing points for visualization
    
    for i in range(len(q_px)):
        # Check if ray crosses edge from vertex j to vertex i
        edge_crosses_y = (q_py[i] > py_test) != (q_py[j] > py_test)
        
        if edge_crosses_y:
            # Calculate x-coordinate where ray crosses edge
            if q_py[j] != q_py[i]:  # Avoid division by zero
                x_cross = (q_px[j] - q_px[i]) * (py_test - q_py[i]) / (q_py[j] - q_py[i]) + q_px[i]
                
                if px_test < x_cross:  # Crossing is to the right of point
                    in_polygon = not in_polygon
                    crossings.append((x_cross, py_test))
        
        j = i
    
    return in_polygon, crossings

# Systematically find points with 0, 2, and 3 crossings
# We'll search in a grid pattern to find good examples

def find_point_with_crossings(target_count, x_min, x_max, y_min, y_max, step=0.02):
    """Search for a point with a specific number of crossings using a grid"""
    x_range = [x_min + i * step for i in range(int((x_max - x_min) / step) + 1)]
    y_range = [y_min + i * step for i in range(int((y_max - y_min) / step) + 1)]
    
    for y in y_range:
        for x in x_range:
            is_inside, crossings = pip(x, y, px, py)
            if len(crossings) == target_count:
                return ((x, y), is_inside, crossings, target_count)
    return None

# First, try some manually selected positions that should work
# Based on the tree shape, we know approximate positions

# Test a few strategic points first
test_candidates = {
    0: [(1.8, 1.3), (1.9, 1.4), (2.0, 1.2), (1.7, 1.5)],  # Far right/up - should be 0
    1: [(1.35, 0.15), (1.4, 0.2), (1.3, 0.1)],  # Edge case - might cross 1
    2: [(1.2, 0.3), (1.25, 0.35), (1.3, 0.4), (1.15, 0.25)],  # Lower left - might cross 2
    3: [(1.52, 0.65), (1.55, 0.7), (1.5, 0.6), (1.53, 0.68)],  # Inside tree - should cross odd number
    4: [(1.5, 0.75), (1.48, 0.8), (1.52, 0.78)]  # Higher inside - might cross 4 or 5
}

examples_found = {0: None, 1: None, 2: None, 3: None, 4: None}

# Try manual candidates first
for count, candidates in test_candidates.items():
    for pt in candidates:
        is_inside, crossings = pip(pt[0], pt[1], px, py)
        if len(crossings) == count:
            examples_found[count] = (pt, is_inside, crossings, count)
            break

# If not found, do grid search
if examples_found[0] is None:
    examples_found[0] = find_point_with_crossings(0, 1.7, 2.1, 1.1, 1.6, 0.05)

if examples_found[2] is None:
    examples_found[2] = find_point_with_crossings(2, 1.1, 1.4, 0.2, 0.5, 0.05)

if examples_found[1] is None:
    examples_found[1] = find_point_with_crossings(1, 1.2, 1.5, 0.1, 0.3, 0.05)

if examples_found[3] is None:
    examples_found[3] = find_point_with_crossings(3, 1.4, 1.7, 0.5, 0.8, 0.05)

if examples_found[4] is None:
    examples_found[4] = find_point_with_crossings(4, 1.45, 1.6, 0.7, 0.9, 0.05)

# If still not found, try even broader search
if examples_found[2] is None:
    # Try points that are to the left of the tree at various heights
    for y in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for x in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]:
            is_inside, crossings = pip(x, y, px, py)
            if len(crossings) == 2:
                examples_found[2] = ((x, y), is_inside, crossings, 2)
                break
        if examples_found[2]:
            break

if examples_found[1] is None:
    # Try points that might cross exactly 1 edge
    for y in [0.1, 0.15, 0.2, 0.25]:
        for x in [1.3, 1.35, 1.4, 1.45]:
            is_inside, crossings = pip(x, y, px, py)
            if len(crossings) == 1:
                examples_found[1] = ((x, y), is_inside, crossings, 1)
                break
        if examples_found[1]:
            break

if examples_found[3] is None:
    # Try points clearly inside the tree
    for y in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        for x in [1.45, 1.5, 1.55, 1.6]:
            is_inside, crossings = pip(x, y, px, py)
            if len(crossings) == 3:
                examples_found[3] = ((x, y), is_inside, crossings, 3)
                break
        if examples_found[3]:
            break

if examples_found[4] is None:
    # Try points higher inside the tree (might cross 4 or 5)
    for y in [0.7, 0.75, 0.8, 0.85]:
        for x in [1.45, 1.5, 1.55]:
            is_inside, crossings = pip(x, y, px, py)
            if len(crossings) == 4:
                examples_found[4] = ((x, y), is_inside, crossings, 4)
                break
        if examples_found[4]:
            break

example_0 = examples_found[0]
example_1 = examples_found[1]
example_2 = examples_found[2]
example_3 = examples_found[3]
example_4 = examples_found[4]

# Build examples list - prioritize 0, 1, 2, 3 crossings (or 4 if we can't find 1)
examples = []
if example_0:
    examples.append(example_0)
if example_1:
    examples.append(example_1)
if example_2:
    examples.append(example_2)
if example_3:
    examples.append(example_3)
# Add 4th example if we have it and don't have 1
if example_4 and example_1 is None:
    examples.append(example_4)
elif example_4 and len(examples) < 4:
    examples.append(example_4)

# Filter out None values and debug
examples = [ex for ex in examples if ex is not None]
print(f"\nDebug - Found examples:")
for ex in examples:
    print(f"  {ex[3]} crossings at {ex[0]}")

# Sort by crossing count
examples.sort(key=lambda x: x[3])

# Create visualization with 2x2 grid (4 subplots)
fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(16, 12))

# Helper function to plot a single ray casting example
def plot_ray_casting(ax, pt, is_inside, crossings, count, title_suffix=""):
    ax.plot(px_plot, py_plot, 'g-', linewidth=2, label='Tree Polygon')
    ax.fill(px_plot, py_plot, 'g', alpha=0.2)
    ax.scatter(px, py, color='red', s=30, zorder=3, alpha=0.6)
    
    # Draw the test point
    ax.scatter([pt[0]], [pt[1]], color='blue', s=200, 
               marker='o', zorder=10, label='Test Point', edgecolors='black', linewidths=2)
    
    # Draw horizontal ray
    ray_end = max(px) + 0.3
    ax.plot([pt[0], ray_end], [pt[1], pt[1]], 
            'r--', linewidth=2, alpha=0.7, label='Ray (shoots right)')
    
    # Mark crossing points with numbers
    if crossings:
        for idx, (x_cross, y_cross) in enumerate(crossings, 1):
            ax.scatter([x_cross], [y_cross], color='orange', s=200, marker='X', 
                      zorder=9, edgecolors='red', linewidths=2)
            # Add number label
            ax.annotate(f'{idx}', (x_cross, y_cross), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=12, fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add annotation
    result_text = "INSIDE" if is_inside else "OUTSIDE"
    color = 'green' if is_inside else 'red'
    parity = "ODD" if count % 2 == 1 else "EVEN"
    ax.text(0.5, 0.95, f'Result: {result_text}\nCrossings: {count} ({parity})',
            transform=ax.transAxes, fontsize=13, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            verticalalignment='top', horizontalalignment='center')
    
    ax.set_title(f"{title_suffix}\nPoint ({pt[0]:.2f}, {pt[1]:.2f})", fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

# Plot the four examples in 2x2 grid
axes = [ax3, ax4, ax5, ax6]
titles = ["0 Crossings (OUTSIDE)", "1 Crossing (OUTSIDE)", "2 Crossings (OUTSIDE)", "3 Crossings (INSIDE)"]

# Ensure we have 4 examples - if we're missing some, do comprehensive search
if len(examples) < 4:
    # Comprehensive grid search as last resort
    print("Doing comprehensive search for missing examples...")
    all_results = {}
    
    # Search a wide area around the tree (tree center is ~1.5, 1.0)
    # Search more densely in areas likely to have the crossings we want
    search_ranges = [
        # For 0 crossings: far from tree
        ([(x/10, y/10) for x in range(17, 25) for y in range(12, 18)], 0),
        # For 1 crossing: edge case
        ([(x/10, y/10) for x in range(12, 16) for y in range(1, 4)], 1),
        # For 2 crossings: to the left/right of tree at mid heights
        ([(x/10, y/10) for x in range(10, 15) for y in range(2, 8)], 2),
        # For 3 crossings: inside the tree
        ([(x/10, y/10) for x in range(14, 18) for y in range(5, 10)], 3),
        # For 4 crossings: higher inside the tree
        ([(x/10, y/10) for x in range(14, 17) for y in range(7, 10)], 4),
    ]
    
    for points_list, target_count in search_ranges:
        if target_count in all_results:
            continue
        for x, y in points_list:
            is_inside, crossings = pip(x, y, px, py)
            count = len(crossings)
            if count == target_count:
                all_results[target_count] = ((x, y), is_inside, crossings, count)
                break
    
    # If still missing, do a broader search
    needed = [c for c in [0, 1, 2, 3, 4] if c not in all_results]
    if needed:
        for y in [y/20 for y in range(0, 40, 1)]:  # 0.0 to 2.0
            for x in [x/20 for x in range(20, 50, 1)]:  # 1.0 to 2.5
                is_inside, crossings = pip(x, y, px, py)
                count = len(crossings)
                if count in needed:
                    all_results[count] = ((x, y), is_inside, crossings, count)
                    needed.remove(count)
                if not needed:
                    break
            if not needed:
                break
    
    # Add any we found that we're missing (prioritize 0, 1, 2, 3)
    for count in [0, 1, 2, 3, 4]:
        if count not in [ex[3] for ex in examples if ex] and count in all_results:
            examples.append(all_results[count])
            print(f"  Found {count} crossings example at {all_results[count][0]}")
            if len(examples) >= 4:
                break

# Sort examples by crossing count for better organization
examples.sort(key=lambda x: x[3])

# Ensure we have exactly 4 examples - fill missing slots with any available examples
# Priority: 0, 1, 2, 3 crossings (or use 4 if needed)
while len(examples) < 4:
    # Try to find any example we don't have yet
    for test_count in [0, 1, 2, 3, 4, 5]:
        if test_count not in [ex[3] for ex in examples]:
            # Quick search for this count
            found = None
            for y in [y/20 for y in range(0, 40, 2)]:
                for x in [x/20 for x in range(20, 50, 2)]:
                    is_inside, crossings = pip(x, y, px, py)
                    count = len(crossings)
                    if count == test_count:
                        found = ((x, y), is_inside, crossings, count)
                        break
                if found:
                    break
            if found:
                examples.append(found)
                print(f"  Added {test_count} crossings example at {found[0]}")
                break
    # If we still don't have 4, use a default point
    if len(examples) < 4:
        pt = (1.5, 0.8)
        is_inside, crossings = pip(pt[0], pt[1], px, py)
        count = len(crossings)
        examples.append((pt, is_inside, crossings, count))
        print(f"  Added default example at {pt} with {count} crossings")

# Sort again after adding
examples.sort(key=lambda x: x[3])
# Take first 4
examples = examples[:4]

# Plot each example
for idx, (ax, title) in enumerate(zip(axes, titles)):
    if idx < len(examples):
        pt, is_inside, crossings, count = examples[idx]
        # Update title to match actual count
        if count == 0:
            title = "0 Crossings (OUTSIDE)"
        elif count == 1:
            title = "1 Crossing (OUTSIDE)"
        elif count == 2:
            title = "2 Crossings (OUTSIDE)"
        elif count == 3:
            title = "3 Crossings (INSIDE)"
        elif count == 4:
            title = "4 Crossings (INSIDE)"
        else:
            title = f"{count} Crossings ({'INSIDE' if is_inside else 'OUTSIDE'})"
        plot_ray_casting(ax, pt, is_inside, crossings, count, title)
    else:
        # Final fallback - should not reach here, but just in case
        pt = (1.5, 0.5)
        is_inside, crossings = pip(pt[0], pt[1], px, py)
        count = len(crossings)
        title = f"{count} Crossings ({'INSIDE' if is_inside else 'OUTSIDE'})"
        plot_ray_casting(ax, pt, is_inside, crossings, count, title)

plt.tight_layout()
plt.savefig('ray_casting_visualization.png', dpi=150, bbox_inches='tight')
print("\nRay casting visualization saved as ray_casting_visualization.png")
print(f"\nPoint-In-Polygon Test Results:")
for pt, is_inside, crossings, count in examples[:4]:
    result = "INSIDE" if is_inside else "OUTSIDE"
    parity = "ODD" if count % 2 == 1 else "EVEN"
    print(f"  - Point {pt}: {result} ({count} crossings, {parity})")
print(f"\nRule: ODD crossings = INSIDE, EVEN crossings = OUTSIDE")

# ============================================================================
# OVERLAP DETECTION VISUALIZATION
# ============================================================================

# Transform tree from base shape to position (cx, cy) with rotation
def getPoly(cx, cy, deg, px, py):
    """Transform tree and fill px, py arrays"""
    rad = deg * math.pi / 180.0
    s = math.sin(rad)
    c = math.cos(rad)
    
    for i in range(len(TX)):
        x_rot = TX[i] * c - TY[i] * s
        y_rot = TX[i] * s + TY[i] * c
        px[i] = x_rot + cx
        py[i] = y_rot + cy

# Create two tree configurations
# Tree A: at position (1.0, 1.0), no rotation
px_a = [0] * 15
py_a = [0] * 15
getPoly(1.0, 1.0, 0, px_a, py_a)

# Tree B: overlapping case - close to Tree A
px_b_overlap = [0] * 15
py_b_overlap = [0] * 15
getPoly(1.3, 1.0, 15, px_b_overlap, py_b_overlap)

# Tree B: non-overlapping case - far from Tree A
px_b_no_overlap = [0] * 15
py_b_no_overlap = [0] * 15
getPoly(2.0, 1.0, 30, px_b_no_overlap, py_b_no_overlap)

# Check overlap by testing if any vertex of one tree is inside the other
def check_overlap_vertices(px_a, py_a, px_b, py_b):
    """Check if trees overlap by testing vertices"""
    overlap_found = False
    overlapping_vertex = None
    overlapping_tree = None
    
    # Check if any vertex of Tree A is inside Tree B
    for i in range(15):
        is_inside, crossings = pip(px_a[i], py_a[i], px_b, py_b)
        if is_inside:
            overlap_found = True
            overlapping_vertex = (px_a[i], py_a[i], i)
            overlapping_tree = "A"
            break
    
    # Check if any vertex of Tree B is inside Tree A
    if not overlap_found:
        for i in range(15):
            is_inside, crossings = pip(px_b[i], py_b[i], px_a, py_a)
            if is_inside:
                overlap_found = True
                overlapping_vertex = (px_b[i], py_b[i], i)
                overlapping_tree = "B"
                break
    
    return overlap_found, overlapping_vertex, overlapping_tree

# Create visualization
fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16))

# ============================================================================
# TOP-LEFT: Overlapping Trees - Overview
# ============================================================================
ax1.plot(px_a + [px_a[0]], py_a + [py_a[0]], 'g-', linewidth=2, label='Tree A')
ax1.fill(px_a + [px_a[0]], py_a + [py_a[0]], 'g', alpha=0.3)
ax1.plot(px_b_overlap + [px_b_overlap[0]], py_b_overlap + [py_b_overlap[0]], 
         'b-', linewidth=2, label='Tree B')
ax1.fill(px_b_overlap + [px_b_overlap[0]], py_b_overlap + [py_b_overlap[0]], 
         'b', alpha=0.3)
ax1.scatter(px_a, py_a, color='darkgreen', s=30, zorder=3, alpha=0.6)
ax1.scatter(px_b_overlap, py_b_overlap, color='darkblue', s=30, zorder=3, alpha=0.6)

# Find overlapping vertex
overlap, ov_vertex, ov_tree = check_overlap_vertices(px_a, py_a, px_b_overlap, py_b_overlap)
if overlap:
    ax1.scatter([ov_vertex[0]], [ov_vertex[1]], color='red', s=300, 
               marker='*', zorder=10, edgecolors='black', linewidths=2,
               label=f'Overlapping Vertex (Tree {ov_tree})')

ax1.set_title("OVERLAPPING TREES\nTree B overlaps with Tree A", fontsize=14, fontweight='bold', color='red')
ax1.axis('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.text(0.02, 0.98, 'OVERLAP DETECTED!', transform=ax1.transAxes,
         fontsize=16, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
         verticalalignment='top')

# ============================================================================
# TOP-RIGHT: Non-Overlapping Trees - Overview
# ============================================================================
ax2.plot(px_a + [px_a[0]], py_a + [py_a[0]], 'g-', linewidth=2, label='Tree A')
ax2.fill(px_a + [px_a[0]], py_a + [py_a[0]], 'g', alpha=0.3)
ax2.plot(px_b_no_overlap + [px_b_no_overlap[0]], py_b_no_overlap + [py_b_no_overlap[0]], 
         'b-', linewidth=2, label='Tree B')
ax2.fill(px_b_no_overlap + [px_b_no_overlap[0]], py_b_no_overlap + [py_b_no_overlap[0]], 
         'b', alpha=0.3)
ax2.scatter(px_a, py_a, color='darkgreen', s=30, zorder=3, alpha=0.6)
ax2.scatter(px_b_no_overlap, py_b_no_overlap, color='darkblue', s=30, zorder=3, alpha=0.6)

overlap2, _, _ = check_overlap_vertices(px_a, py_a, px_b_no_overlap, py_b_no_overlap)

ax2.set_title("NON-OVERLAPPING TREES\nTrees are separated", fontsize=14, fontweight='bold', color='green')
ax2.axis('equal')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
ax2.text(0.02, 0.98, 'NO OVERLAP', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         verticalalignment='top')

# ============================================================================
# BOTTOM-LEFT: Ray Casting Check - Overlapping Case
# ============================================================================
# Find a vertex of Tree B that's inside Tree A
test_vertex_idx = None
test_vertex_x = None
test_vertex_y = None
test_crossings = None

for i in range(15):
    is_inside, crossings = pip(px_b_overlap[i], py_b_overlap[i], px_a, py_a)
    if is_inside:
        test_vertex_idx = i
        test_vertex_x = px_b_overlap[i]
        test_vertex_y = py_b_overlap[i]
        test_crossings = crossings
        break

if test_vertex_idx is not None:
    # Draw Tree A
    ax3.plot(px_a + [px_a[0]], py_a + [py_a[0]], 'g-', linewidth=2, label='Tree A')
    ax3.fill(px_a + [px_a[0]], py_a + [py_a[0]], 'g', alpha=0.2)
    ax3.scatter(px_a, py_a, color='darkgreen', s=30, zorder=3, alpha=0.5)
    
    # Draw Tree B (lighter)
    ax3.plot(px_b_overlap + [px_b_overlap[0]], py_b_overlap + [py_b_overlap[0]], 
             'b--', linewidth=1.5, alpha=0.5, label='Tree B')
    ax3.scatter(px_b_overlap, py_b_overlap, color='lightblue', s=20, zorder=2, alpha=0.4)
    
    # Highlight the test vertex
    ax3.scatter([test_vertex_x], [test_vertex_y], color='red', s=400, 
               marker='*', zorder=10, edgecolors='black', linewidths=2,
               label=f'Tree B Vertex {test_vertex_idx}')
    
    # Draw ray from test vertex
    ray_end = max(px_a) + 0.5
    ax3.plot([test_vertex_x, ray_end], [test_vertex_y, test_vertex_y], 
             'r--', linewidth=2.5, alpha=0.8, label='Ray (shoots right)')
    
    # Mark crossing points
    if test_crossings:
        for idx, (x_cross, y_cross) in enumerate(test_crossings, 1):
            ax3.scatter([x_cross], [y_cross], color='orange', s=250, marker='X', 
                       zorder=9, edgecolors='red', linewidths=2)
            ax3.annotate(f'{idx}', (x_cross, y_cross), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=14, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    
    # Result annotation
    count = len(test_crossings) if test_crossings else 0
    result_text = "INSIDE" if count % 2 == 1 else "OUTSIDE"
    color = 'green' if count % 2 == 1 else 'red'
    parity = "ODD" if count % 2 == 1 else "EVEN"
    
    ax3.text(0.5, 0.95, f'Vertex of Tree B inside Tree A?\nCrossings: {count} ({parity})\nResult: {result_text} → OVERLAP!', 
             transform=ax3.transAxes, fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
             verticalalignment='top', horizontalalignment='center')
    
    ax3.set_title(f"Ray Casting Check: Overlapping Case\nTesting Tree B vertex {test_vertex_idx} against Tree A", 
                  fontsize=12, fontweight='bold')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)

# ============================================================================
# BOTTOM-RIGHT: Ray Casting Check - Non-Overlapping Case
# ============================================================================
# Test a vertex of Tree B that's outside Tree A
test_vertex_idx2 = 0  # Use first vertex
test_vertex_x2 = px_b_no_overlap[test_vertex_idx2]
test_vertex_y2 = py_b_no_overlap[test_vertex_idx2]
is_inside2, test_crossings2 = pip(test_vertex_x2, test_vertex_y2, px_a, py_a)

# Draw Tree A
ax4.plot(px_a + [px_a[0]], py_a + [py_a[0]], 'g-', linewidth=2, label='Tree A')
ax4.fill(px_a + [px_a[0]], py_a + [py_a[0]], 'g', alpha=0.2)
ax4.scatter(px_a, py_a, color='darkgreen', s=30, zorder=3, alpha=0.5)

# Draw Tree B (lighter)
ax4.plot(px_b_no_overlap + [px_b_no_overlap[0]], py_b_no_overlap + [py_b_no_overlap[0]], 
         'b--', linewidth=1.5, alpha=0.5, label='Tree B')
ax4.scatter(px_b_no_overlap, py_b_no_overlap, color='lightblue', s=20, zorder=2, alpha=0.4)

# Highlight the test vertex
ax4.scatter([test_vertex_x2], [test_vertex_y2], color='blue', s=400, 
           marker='*', zorder=10, edgecolors='black', linewidths=2,
           label=f'Tree B Vertex {test_vertex_idx2}')

# Draw ray from test vertex
ray_end2 = max(px_a) + 0.5
ax4.plot([test_vertex_x2, ray_end2], [test_vertex_y2, test_vertex_y2], 
         'r--', linewidth=2.5, alpha=0.8, label='Ray (shoots right)')

# Mark crossing points (if any)
if test_crossings2:
    for idx, (x_cross, y_cross) in enumerate(test_crossings2, 1):
        ax4.scatter([x_cross], [y_cross], color='orange', s=250, marker='X', 
                   zorder=9, edgecolors='red', linewidths=2)
        ax4.annotate(f'{idx}', (x_cross, y_cross), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=14, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))

# Result annotation
count2 = len(test_crossings2) if test_crossings2 else 0
result_text2 = "INSIDE" if count2 % 2 == 1 else "OUTSIDE"
color2 = 'green' if count2 % 2 == 1 else 'red'
parity2 = "ODD" if count2 % 2 == 1 else "EVEN"

ax4.text(0.5, 0.95, f'Vertex of Tree B inside Tree A?\nCrossings: {count2} ({parity2})\nResult: {result_text2} → NO OVERLAP', 
         transform=ax4.transAxes, fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor=color2, alpha=0.3),
         verticalalignment='top', horizontalalignment='center')

ax4.set_title(f"Ray Casting Check: Non-Overlapping Case\nTesting Tree B vertex {test_vertex_idx2} against Tree A", 
              fontsize=12, fontweight='bold')
ax4.axis('equal')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('overlap_detection_visualization.png', dpi=150, bbox_inches='tight')
print("\nOverlap detection visualization saved as overlap_detection_visualization.png")
print("\n" + "="*60)
print("HOW RAY CASTING DETECTS OVERLAP:")
print("="*60)
print("\n1. For each vertex (corner) of Tree B:")
print("   - Use ray casting to check if it's inside Tree A")
print("   - If ODD crossings → vertex is INSIDE → OVERLAP!")
print("   - If EVEN crossings → vertex is OUTSIDE → keep checking")
print("\n2. Also check vertices of Tree A inside Tree B")
print("\n3. If ANY vertex is inside the other tree → OVERLAP")
print("\n4. The visualization shows:")
print("   - Top-left: Overlapping trees (red star = overlapping vertex)")
print("   - Top-right: Non-overlapping trees")
print("   - Bottom-left: Ray casting check for overlapping case")
print("   - Bottom-right: Ray casting check for non-overlapping case")

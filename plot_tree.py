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

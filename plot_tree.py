import matplotlib.pyplot as plt

# The Constants from the C++ code
TX = [0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075, -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125]
TY = [0.8, 0.5, 0.5, 0.25, 0.25, 0, 0, -0.2, -0.2, 0, 0, 0.25, 0.25, 0.5, 0.5]

# Close the loop for plotting (connect last point back to first)
TX.append(TX[0])
TY.append(TY[0])

# Plotting
plt.figure(figsize=(6, 8))
plt.plot(TX, TY, 'g-', linewidth=2, label='Tree Polygon')  # 'g-' means green line
plt.fill(TX, TY, 'g', alpha=0.3)  # Fill with light green
plt.scatter(TX, TY, color='red', s=50, zorder=5)  # Mark the vertices with red dots

# Label the vertices
for i in range(len(TX)-1):
    plt.annotate(f'{i}', (TX[i], TY[i]), xytext=(5, 5), textcoords='offset points')

plt.title("The 15-Vertex Christmas Tree")
plt.axis('equal')  # Ensure aspect ratio is 1:1 so it doesn't look squashed
plt.grid(True)
plt.savefig('tree_visualization.png')
print("Tree visualization saved as tree_visualization.png")

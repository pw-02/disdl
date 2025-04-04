import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Create custom handles using invisible lines (for spacing purposes) and labels
handle_t = Line2D([0], [0], color="none", label="T = TensorSocket")  # Invisible handle
handle_c = Line2D([0], [0], color="none", label="C = CoorDL")        # Invisible handle
handle_d = Line2D([0], [0], color="none", label="D = DisDL")          # Invisible handle

# Create a plot to show the legend
fig, ax = plt.subplots()

# Add some plot elements (just for the context)
ax.plot([0, 1], [0, 1], label="Line Plot")
ax.bar([0, 1], [1, 2], label="Bar Plot")

# Manually define the handles and labels for the legend
handles = [handle_t, handle_c, handle_d]
labels = ["T = TensorSocket", "C = CoorDL", "D = DisDL"]

# Display the legend with text-only handles
ax.legend(
    handles=handles, 
    labels=labels, 
    loc="upper left", 
    fontsize=10,
    handlelength=0,  # Set handle length to 0 to remove extra space
    handleheight=0,  # Set handle height to 0 to remove the empty space
    borderpad=0.2,   # Adjust border padding for compactness
    columnspacing=1  # Adjust the spacing between legend columns
)

# Show the plot
plt.show()

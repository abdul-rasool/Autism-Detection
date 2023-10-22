import networkx as nx
import numpy as np
from skimage import io, color, util
import matplotlib.pyplot as plt


image_path="F:\extramyplot2.png"
image = io.imread(image_path)


# Checking if the image has 4 channels
if image.shape[2] == 4:
    image = image[:, :, :3]

# Converting to grayscale
image = color.rgb2gray(image)

# Downsampling the image to reduce computation
image = util.img_as_ubyte(image[::10, ::10])

# Creating an empty graph
G = nx.Graph()

# Adding a node for each pixel in the image
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        G.add_node((i, j), intensity=image[i, j])

# Connecting each node with its neighbors
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if i > 0:
            G.add_edge((i, j), (i-1, j))
        if j > 0:
            G.add_edge((i, j), (i, j-1))

# Defining a position mapping
pos = {(x, y):(y, -x) for (x, y), d in G.nodes(data=True)}

# Plotting the image
plt.imshow(image, cmap='gray')

# Drawing a subset of nodes
subset_nodes = [(x, y) for x, y in G.nodes() if 0 <= x < 10 and 0 <= y < 10] # Adjust range as per your requirements
nx.draw_networkx_nodes(G, pos, nodelist=subset_nodes, node_size=2, node_color='r')

# Drawing a subset of edges
subset_edges = [edge for edge in G.edges() if edge[0] in subset_nodes and edge[1] in subset_nodes]
nx.draw_networkx_edges(G, pos, edgelist=subset_edges, edge_color='blue')

# Showing the plot
plt.show()
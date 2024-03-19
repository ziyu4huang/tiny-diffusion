
import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

def draw_line_dino(ax):

    # Load the data
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    # Assuming df['x'] and df['y'] are Series containing the x and y coordinates of the polygon's vertices
    # x_vertices = df['x'].tolist()
    # y_vertices = df['y'].tolist()
    # # Combine the x and y vertices into a list of tuples
    # polygon_vertices = list(zip(x_vertices, y_vertices))
    polygon_vertices = list(zip(df['x'], df['y']))

    # Create a polygon patch
    #polygon = Polygon(polygon_vertices, closed=True, edgecolor='r', facecolor='none')
    polygon = Polygon(polygon_vertices, closed=False, edgecolor='r', facecolor='none')

    # Add the polygon patch to the axes
    ax.add_patch(polygon)

    # Set the limits of the plot based on the polygon's vertices
    # ax.set_xlim([min(x_vertices), max(x_vertices)])
    # ax.set_ylim([min(y_vertices), max(y_vertices)])

    ax.set_xlim([min(df['x']), max(df['x'])])
    ax.set_ylim([min(df['y']), max(df['y'])])

    ax.grid(True)
    ax.set_title('Dino Dataset Polygon')

def draw_dot(ax, n=0, title=""):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    if n == 0:
        x = df['x']
        y = df['y']
    else:
        print(f"dot random sample with n={n}")
        rng = np.random.default_rng(42)
        ix = rng.integers(0, len(df), n)
        x = df["x"].iloc[ix]
        y = df["y"].iloc[ix]
        x += rng.normal(size=len(x)) * 0.15
        y += rng.normal(size=len(y)) * 0.15

        x = (x/54 - 1) * 4
        y = (y/48 - 1) * 4

    ax.scatter(x, y, alpha=0.5)
    ax.grid(True)
    ax.set_title(f'Dino Dataset Dot {title}')

if __name__ == "__main__":

    # Setup the plot
    fig, axn = plt.subplots(1, 3, figsize=(20, 10))  # Adjust size as needed
    #axs = axs.flatten()  # Flatten to easily index them

    draw_dot(axn[0], n=0, title="original")
    draw_dot(axn[1], n=200, title="n=200")
    draw_dot(axn[2], n=8000, title="n=8000")
    #draw_line_dino(ax2)

    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # Set a suptitle for the figure (spans the entire window)
    title = 'Dino Dataset Visualization'
    # fig.suptitle(title)
    # Set the window title (appears in the title bar of the window)
    fig.canvas.manager.set_window_title(title)

    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereographic_projection(x, y, z):
    """Applies stereographic projection from the sphere to the plane."""
    return x / (1 - z), y / (1 - z)

def rotate_about_axis(x, y, z, axis, angle):
    """Rotates points around a specified axis by a given angle using Rodrigues' rotation formula."""
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    rot_matrix = np.array([[a*a + b*b - c*c - d*d, 2*(b*c + a*d), 2*(b*d - a*c)],
                           [2*(b*c - a*d), a*a + c*c - b*b - d*d, 2*(c*d + a*b)],
                           [2*(b*d + a*c), 2*(c*d - a*b), a*a + d*d - b*b - c*c]])
    return np.dot(rot_matrix, np.array([x, y, z]))

def plot_great_circles():
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # Define a single great circle (equatorial)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(theta)

    # Plot the sphere
    phi = np.linspace(0, np.pi, 50)
    theta_grid = np.linspace(0, 2 * np.pi, 50)
    phi, theta_grid = np.meshgrid(phi, theta_grid)
    xs = np.sin(phi) * np.cos(theta_grid)
    ys = np.sin(phi) * np.sin(theta_grid)
    zs = np.cos(phi)
    ax1.plot_surface(xs, ys, zs, color='lightblue', alpha=0.3, edgecolor='none')

    # Define great circles at different angles
    colors = ['r', 'g', 'b', 'm', 'orange']
    angles = np.linspace(0, np.pi, len(colors), endpoint=False)
    axes = [[1, 0, 0], [0, 1, 0]]  # Rotate around x and y axes

    projected_x, projected_y = [], []
    
    for i, color in enumerate(colors):
        axis = axes[i % 2]  # Alternate rotation between x and y axes
        angle = angles[i]
        xr, yr, zr = rotate_about_axis(x, y, z, axis, angle)
        
        # Plot on sphere with color coding
        ax1.plot(xr, yr, zr, color=color, linewidth=2)

        # Project and collect data for autoscaling
        xp, yp = stereographic_projection(xr, yr, zr)
        projected_x.extend(xp)
        projected_y.extend(yp)

        # Plot projection with the same color
        ax2.plot(xp, yp, color=color, linewidth=2)

    # Adjust axis limits dynamically for better visibility
    ax2.set_xlim(min(projected_x) * 1.1, max(projected_x) * 1.1)
    ax2.set_ylim(min(projected_y) * 1.1, max(projected_y) * 1.1)

    # Titles
    ax1.set_title('Great Circles on Sphere')
    ax2.set_title('Stereographic Projection of Great Circles')
    ax2.set_aspect('equal', adjustable='datalim')

    plt.show()
    plt.savefig('gc_demo.png')

plot_great_circles()

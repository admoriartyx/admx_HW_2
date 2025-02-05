# Problem 2

# Part a
# Goal is to show transformation is conformal, meaning angles are perserved in the mapping.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stereo_proj(x, y, z):
    denom = 1-z
    return x/denom, y/denom

# The above was just a code implementation of the formula already provided. Now for a plotting function.

def plot_stereo(theta, phi): # The xp, yp, zp are coordinates of point P on sphere surface
    xp = np.sin(theta) * np.cos(phi) 
    yp = np.sin(theta) * np.sin(phi)
    zp = np.cos(theta)
    P = np.array([xp, yp, zp])
    e_theta = np.array([ np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_phi_unit = e_phi / np.sin(theta)
    
    # Now we can think about generating curves along the paths above.
    # For that, we need a parametrization.
    
    t = np.linspace(-0.1, 0.1, 100)

    c1 = np.array([P + i * e_theta for i in t])
    c1 = c1 / np.linalg.norm(c1, axis=1)[:, np.newaxis]

    # generate curve along e_phi_unit
    c2 = np.array([P + i * e_phi_unit for i in t])
    c2 = c2 / np.linalg.norm(c2, axis=1)[:, np.newaxis]

    # now for projections
    c1_proj = stereo_proj(c1[:, 0], c1[:, 1], c1[:, 2])
    c2_proj = stereo_proj(c2[:, 0], c2[:, 1], c2[:, 2])

    fig = plt.figure(figsize=(10, 5))
    plot_3D = fig.add_subplot(121, projection='3d')
    plot_3D.plot(c1[:, 0], c1[:, 1], c1[:, 2], label='Curve 1')
    plot_3D.plot(c2[:, 0], c2[:, 1], c2[:, 2], label='Curve 2')
    plot_3D.scatter(P[0], P[1], P[2], color='red', s=50, label='Point P')
    plot_3D.quiver(P[0], P[1], P[2], e_theta[0], e_theta[1], e_theta[2], color='blue', length=0.1, label='Tangent 1')
    plot_3D.quiver(P[0], P[1], P[2], e_phi_unit[0], e_phi_unit[1], e_phi_unit[2], color='green', length=0.1, label='Tangent 2')
    plot_3D.set_title('Unit Sphere')
    plot_3D.legend()
    plot_3D.set_xlabel('x')
    plot_3D.set_ylabel('y')
    plot_3D.set_zlabel('z')


    plot_2D = fig.add_subplot(122)
    plot_2D.plot(c1_proj[0], c1_proj[1], label='Curve 1 Projection')
    plot_2D.plot(c2_proj[0], c2_proj[1], label='Curve 2 Projection')
    plot_2D.scatter(*stereo_proj(P[0], P[1], P[2]), color='green', s=50, label='Projected P')

    scale = 0.5
    e_theta_dS = np.array([-0.471, -0.471]) * scale
    e_phi_dS = np.array([-0.544, 0.544]) * scale
    plot_2D.quiver(*stereo_proj(P[0], P[1], P[2]), e_theta_dS[0], e_theta_dS[1], color='blue', scale=1, scale_units='xy', angles='xy', label='Projected Tangent 1')
    plot_2D.quiver(*stereo_proj(P[0], P[1], P[2]), e_phi_dS[0], e_phi_dS[1], color='black', scale=1, scale_units='xy', angles='xy', label='Projected Tangent 2')
    plot_2D.set_title('Stereographic Projection')
    plot_2D.legend()
    plot_2D.set_label('x')
    plot_2D.set_label('y')
    plot_2D.grid(True)
    plot_2D.axis('equal')
    plot_3D.text(P[0], P[1], P[2], '90', color='black')
    plot_2D.text(stereo_proj(P[0], P[1], P[2])[0], stereo_proj(P[0], P[1], P[2])[1], '90', color='black')
    plt.tight_layout()
    plt.show()
    plt.savefig('Stereographic_Projection.png')

#plot_stereo(2*np.pi/3, np.pi/4)

# The preservation of the angles in the projection shown in the .png demonstrates conformality.

# Part b
# Goal of part b is to plot the geodesics of the sphere and notice "great circle" properties.

def rotate_about_axis(x, y, z, axis, angle):
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
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(theta)

    phi = np.linspace(0, np.pi, 50)
    theta_grid = np.linspace(0, 2 * np.pi, 50)
    phi, theta_grid = np.meshgrid(phi, theta_grid)
    xs = np.sin(phi) * np.cos(theta_grid)
    ys = np.sin(phi) * np.sin(theta_grid)
    zs = np.cos(phi)
    ax1.plot_surface(xs, ys, zs, color='lightblue', alpha=0.3, edgecolor='none')

    # I want all the great circles to be at different angular orientations.
    colors = ['r', 'g', 'b', 'm', 'orange']
    angles = np.linspace(0, np.pi, len(colors), endpoint=False)
    axes = [[1, 0, 0], [0, 1, 0]] 

    projected_x, projected_y = [], []
    
    for i, color in enumerate(colors):
        axis = axes[i % 2]  # Alternating rotation between x and y axes for uniqueness
        angle = angles[i]
        xr, yr, zr = rotate_about_axis(x, y, z, axis, angle)
        
        # color coding the great circles and their projections as well
        ax1.plot(xr, yr, zr, color=color, linewidth=2)
        xp, yp = stereo_proj(xr, yr, zr)
        projected_x.extend(xp)
        projected_y.extend(yp)
        ax2.plot(xp, yp, color=color, linewidth=2)

    # Plot axes were very off for some reason so had to manually edit
    ax2.set_xlim(min(projected_x) * 1.1, max(projected_x) * 1.1)
    ax2.set_ylim(min(projected_y) * 1.1, max(projected_y) * 1.1)

    ax1.set_title('Great Circles on Sphere')
    ax2.set_title('Stereographic Projection of Great Circles')
    ax2.set_aspect('equal', adjustable='datalim')

    plt.show()
    plt.savefig('great_circles.png')

#plot_great_circles()


# Part c









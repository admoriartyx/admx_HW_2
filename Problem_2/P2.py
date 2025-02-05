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

plot_stereo(2*np.pi/3, np.pi/4)

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

plot_great_circles()


# Part c
# Pulling the old function from Problem 1

def spherical_to_cartesian_new(theta, phi):
    y = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    z = np.array([-np.sin(phi), np.cos(phi), 0])
    return y, z

def parallel_transport_projection(theta0=np.pi/4):
    alpha = 1.0
    beta = 0.0
    n_mag = 1.0

    phis = np.linspace(0, 2*np.pi, 30)
    thetas = theta0 * np.ones_like(phis)

    x_sphere = np.sin(theta0) * np.cos(phis)
    y_sphere = np.sin(theta0) * np.sin(phis)
    z_sphere = np.cos(theta0) * np.ones_like(phis)
    
    delta = 2 * np.pi * (1 - np.cos(theta0))
    V_theta = alpha * np.cos(delta * phis/(2*np.pi))
    V_phi = alpha * np.sin(delta * phis/(2*np.pi))

    x_proj, y_proj = stereo_proj(x_sphere, y_sphere, z_sphere)
    
    Vx_proj, Vy_proj = [], []
    for i in range(len(phis)):
        e_theta, e_phi = spherical_to_cartesian_new(theta0, phis[i])
        vec_3d = V_theta[i]*e_theta + V_phi[i]*e_phi
        z = z_sphere[i]
        J = np.array([[1/(1-z), 0, x_sphere[i]/(1-z)**2],
                      [0, 1/(1-z), y_sphere[i]/(1-z)**2]])
        vec_proj = J @ vec_3d
        Vx_proj.append(vec_proj[0])
        Vy_proj.append(vec_proj[1])

    fig = plt.figure(figsize=(12, 6))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_sphere, y_sphere, z_sphere, 'b-', label=f'θ = {theta0:.2f} Path')
    ax1.quiver(x_sphere[::3], y_sphere[::3], z_sphere[::3], 
               V_theta[::3]*e_theta[0], V_theta[::3]*e_theta[1], V_theta[::3]*e_theta[2],
               color='red', length=0.1, label='Transported Vector')
    ax1.set_title(f'3D Parallel Transport\nLatitude θ={theta0:.2f}')
    ax1.legend()
    
    # stereographic projection plot
    ax2 = fig.add_subplot(122)
    ax2.plot(x_proj, y_proj, 'b-', label='Projected Path')
    ax2.quiver(x_proj[::3], y_proj[::3], 
               np.array(Vx_proj)[::3], np.array(Vy_proj)[::3],
               color='red', scale=15, width=0.003, label='Projected Vectors')
    ax2.set_title(f'Stereographic Projection\nRotation: {delta/np.pi:.2f}π')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('Parallel_transport_projection_{theta0:.2f}.png')
    plt.show()

# create plots for different latitudes


for theta0 in [np.pi/6, np.pi/4, np.pi/3]:
    parallel_transport_projection(theta0)

# Part d

# Must check if inner products are conserved in the projection.
# Immediately based on the hint I believe the inner products will be conserved
# because the conformality of the stereographic projection maintains angles. 
# Since the inner product is dependent upon the angle between two vectors, the inner product
# will also be conserved.

# However, the problem still asks for the plots so lets proceed.

def projection_jacobian(x, y, z):
    denom = 1 - z
    J = np.array([[1/denom, 0, x/(denom**2)],
                  [0, 1/denom, y/(denom**2)]])
    return J


num_points = 50
thetas = np.linspace(0, np.pi, num_points)
phis = np.zeros_like(thetas)

results = []
for theta in thetas:
    x = np.sin(theta) * np.cos(0)
    y = np.sin(theta) * np.sin(0)
    z = np.cos(theta)
    
    e_theta = np.array([np.cos(theta), 0, -np.sin(theta)])
    e_phi = np.array([0, 1, 0])
    J = projection_jacobian(x, y, z)
    
    v_proj = J @ e_theta
    w_proj = J @ e_phi
    
    orig_ip = np.dot(e_theta, e_phi)
    proj_ip = np.dot(v_proj, w_proj)
    
    # Note that the first dot product should always be 0 b/c bases vectors.
    conformal_factor = 1/(1 - z)
    
    results.append({
        'theta': theta,
        'z': z,
        'orig_ip': orig_ip,
        'proj_ip': proj_ip,
        'lambda_sq': conformal_factor**2
    })

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(111)
zs = [r['z'] for r in results]
proj_ips = [r['proj_ip'] for r in results]
ax1.scatter(zs, proj_ips, c='r', label='Projected IP')
ax1.plot(zs, [0]*len(zs), 'b--', label='Original IP')
ax1.set_xlabel('z-coordinate on Sphere')
ax1.set_ylabel('Inner Product')
ax1.set_title('Inner Product After Projection\n(Orthogonal Vectors)')
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('Inner_products.png')

# Our plot agrees with the conceptual analysis of the problem! The inner products agree.

# Part f (should be part 3)
# The holonomy of the unit sphere is unchanged after stereographic projection of parallel transport.
# This is in part due to the conformal nature of the stereographic projection. Parallel transport
# is also preserved by the conformality of the projection.







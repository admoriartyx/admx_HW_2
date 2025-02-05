# Problem 1 .py File
# Part a

# The first part of the problem asks for a point on the unit sphere to be defined in 3 different
# coordinate systems. I will write them out below.

# Spherical Coordinates: (1, theta, phi)
# Cartesian Coordinates: (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
# Cylindrical Coordinates: (sin(theta), phi, cos(phi))

# Now we are asked to write the spherical and cylindrical basis vectors in terms of the Cartesian ones.
# I cannot append the ^ symbol directly to characters so just assume x,y,z are their own basis vectors.

# For Spherical Coordinate basis vectors:
# e_r = sin(theta)cos(phi)x + sin(theta)sin(phi)y + cos(theta)z
# e_theta = cos(theta)cos(phi)x + cos(theta)sin(phi)y - sin(theta)z
# e_phi = -sin(phi)x + cos(phi)y

# Now for cylindrical coordinates:
# e_rho = cos(phi)x + sin(phi)y
# e_psi = -sin(phi)x + cos(phi)y
# e_z = z

# Now it is time to write an actual python script that transforms between these coordinate systems.

import numpy as np
import matplotlib.pyplot as plt

# I will write a function for each conversion of bases vectors.

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return (x, y, z)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return (r, theta, phi)

def cylindrical_to_cartesian(rho, psi, z):
    x = rho * np.cos(psi)
    y = rho * np.sin(psi)
    return (x, y, z)

def cartesian_to_cylindrical(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    psi = np.arctan2(y, x)
    return (rho, psi, z)

# Instead of writing two more functions that directly convert between spherical and cylindrical,
# the functions I have already written allow for conversion between the two bases by using the 
# cartesian bases as a medium. In other words, any of the bases can be reached from any of the other two 
# by applying the functions above iteratively, as opposed to writing new functions entirely.


# Part b

# Part b of problem 1 asks for a graph of a sphere with locally orthongonal coodinate systems on the surface.

from mpl_toolkits.mplot3d import Axes3D

def unit_sphere_ortho_basis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.6)
    
    phi_vec, theta_vec = phi[::10, ::10], theta[::10, ::10]
    x_vec, y_vec, z_vec = x[::10, ::10], y[::10, ::10], z[::10, ::10]
    
    er = np.array([np.sin(theta_vec) * np.cos(phi_vec), np.sin(theta_vec) * np.sin(phi_vec), np.cos(theta_vec)])
    etheta = np.array([np.cos(theta_vec) * np.cos(phi_vec), np.cos(theta_vec) * np.sin(phi_vec), -np.sin(theta_vec)])
    ephi = np.array([-np.sin(phi_vec), np.cos(phi_vec), np.zeros_like(phi_vec)])

    er /= np.linalg.norm(er, axis=0)
    etheta /= np.linalg.norm(etheta, axis=0)
    ephi /= np.linalg.norm(ephi, axis=0)

    quiver_scale = 0.1
    ax.quiver(x_vec, y_vec, z_vec, er[0], er[1], er[2], color='red', length=quiver_scale, normalize=True)
    ax.quiver(x_vec, y_vec, z_vec, etheta[0], etheta[1], etheta[2], color='green', length=quiver_scale, normalize=True)
    ax.quiver(x_vec, y_vec, z_vec, ephi[0], ephi[1], ephi[2], color='yellow', length=quiver_scale, normalize=True)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    plt.savefig('3D_Sphere_Ortho_basis_vectors.png')

unit_sphere_ortho_basis()

# Part c
# You cannot plot a unit sphere in referencing only the spherical coordinate bases because
# the orientations of the basis vectors change with respect to the point. This would create an awkward
# sort of recursion while trying to map the sphere.

# Part d

def local_coordinates(f, x_range, y_range, step=0.5):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='b', edgecolor='k')
    
    # Implementing the hint
    fz_x, fz_y = np.gradient(Z, axis=(1, 0))
    norms = np.sqrt(fz_x**2 + fz_y**2 + 1)
    fz_x /= norms
    fz_y /= norms
    fz_z = -1 / norms
    quiver_scale = 0.2
    ax.quiver(X, Y, Z, fz_x, fz_y, fz_z, color='r', length=quiver_scale, normalize=True)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.savefig('local_coordinates.png')

# Part e

# Start off with some initial parameters following the lecture notes

theta0 = np.pi / 5
alpha = 0.0
beta = 1 / np.sin(theta0)

n = 10
thetas = np.linspace(theta0, np.pi/2, n)
phis = np.zeros_like(thetas)

# initialize cartesian coordinates
x = np.sin(thetas)*np.cos(phis)
y = np.sin(thetas)*np.sin(phis)
z = np.cos(thetas)

# now beginning parallel transport: focus vector components in spherical basis
Vtheta = alpha*np.ones_like(thetas)
Vphi = beta*np.sin(thetas) / np.sin(thetas)
Vx = Vtheta*np.cos(thetas)*np.cos(phis) + Vphi*(-np.sin(thetas)*np.sin(phis))
Vy = Vtheta*np.cos(thetas)*np.sin(phis) + Vphi*(np.sin(thetas)*np.cos(phis))
Vz = Vtheta *(-np.sin(thetas))

# now for plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.6)

# now to account for transported vectors being appended to sphere

scale = 0.4
ax.quiver(x, y, z, scale*Vx, scale*Vy, scale*Vz, color='red', label='Parallel Transport Vector', normalize=False )
ax.plot(x, y, z, 'b-', linewidth=2, label='Transport path')
ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
ax.set_title(f'Parallel Transport from θ={theta0:.2f} to Equator')
ax.legend()
ax.view_init(elev=20, azim=-45)  
plt.show()
plt.savefig('parallel_trans_vec.png')

# Part f

# Process here will be similar to before. State initial conditions and then employ transport
# trajectory. This time, however, we have to take the change in vector magnitude into account.

theta0 = np.pi / 4
alpha = 1
beta = 0
v_mag = 1

phis = np.linspace(0, 2*np.pi, 30)
thetas = theta0 * np.ones_like(phis)
x = np.sin(theta0) * np.cos(phis)
y = np.sin(theta0) * np.sin(phis)
z = np.cos(theta0) * np.ones_like(phis)

rot_angle = 2 * np.pi * (1 - np.cos(theta0))
Vtheta = alpha * np.cos(rot_angle * phis/(2*np.pi)) - beta * np.sin(rot_angle * phis/(2*np.pi))
Vphi = alpha * np.sin(rot_angle * phis/(2*np.pi)) + beta * np.cos(rot_angle * phis/(2*np.pi))

Vx, Vy, Vz = [], [], []

# Must edit our original function to output only two values. Assume unit sphere condition.
def spherical_to_cartesian_new(theta, phi):
    y = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    z = np.array([-np.sin(phi), np.cos(phi), 0])
    return y, z

for i in range(len(phis)):
    e_theta, e_phi = spherical_to_cartesian_new(theta0, phis[i])
    Vcart = Vtheta[i]*e_theta + Vphi[i]*e_phi
    Vx.append(Vcart[0])
    Vy.append(Vcart[1])
    Vz.append(Vcart[2])

Vx, Vy, Vz = np.array(Vx), np.array(Vy), np.array(Vz)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(u.size), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.6)
ax.plot(x, y, z, 'b-', linewidth=1.5, label=f'θ = {theta0:.2f} Path')
scale = 0.5
ax.quiver(x[::2], y[::2], z[::2], 
          scale*Vx[::2], scale*Vy[::2], scale*Vz[::2],
          color='red', label='Parallel Transported Vector',
          arrow_length_ratio=0.15, linewidth=1)

ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
ax.set_title(f'Parallel Transport Along Latitude θ={theta0:.2f}')
ax.legend()
ax.view_init(elev=30, azim=-45)
plt.savefig('P1_partf_transport.png')
plt.show()

# Part g (final portion of Problem 1)
# This should be relatively easy, just need to account for new theta0 values and exploit difference

thetas = np.linspace(0, np.pi, 1000)
alpha = 1
beta = 0
inner_prods = []

for theta0 in thetas:
    rot_angle = 2*np.pi*(1-np.cos(theta0))
    final_theta_vec = alpha*np.cos(rot_angle) - beta*np.sin(rot_angle)
    final_phi_vec = alpha*np.sin(rot_angle) + beta*np.cos(rot_angle)

    ip = alpha*final_theta_vec + beta*final_phi_vec
    inner_prods.append(ip)

# plot inner product vs. θ0
plt.figure(figsize=(8, 6))
plt.plot(thetas, inner_prods, 'r-', linewidth=2)
plt.xlabel('initial latitudes')
plt.ylabel('Holonomy Strength (via inner product)')
plt.title('Holonomy Strength vs. Initial Angle')
plt.grid(True)
plt.show()
plt.savefig('Holonomy_plot.png')

# Problem 2 will appear in a new .py file.
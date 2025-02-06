# THIS IS THE .PY FOR PART H ONLY
# I copy and pasted my code from the rest of the problem, trying to just tweak the small details

# Part a
# Must use mesh.dat file to plot convex hull and Delaunay Triangulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

points = np.loadtxt('mesh.dat', skiprows=1)
hull = ConvexHull(points)
tri = Delaunay(points)

# part b 

# This should be cool. Lifting the triangles into 3D using a set z function.

def z(x, y):
    return x**2 + x*y + y**2

zs = z(points[:, 0], points[:, 1])
coords_3d = np.column_stack((points, zs))

def tri_area_2d(points):
    a, b, c = points
    return 0.5 * np.abs(np.cross(b - a, c - a))

def tri_area_3d(points):
    a, b, c = points
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross)

area_ratios = []
for i in tri.simplices:
    tri_2d = points[i]
    tri_3d = coords_3d[i]

    area_2d = tri_area_2d(tri_2d)
    area_3d = tri_area_3d(tri_3d)
    
    area_ratios.append(area_3d / area_2d)

area_ratios = np.array(area_ratios)

area_ratios_limiter = np.clip(area_ratios, 0, 100)

area_ratios_log = np.log10(area_ratios_limiter + 1e-6)
area_ratios_normalized = (area_ratios_log - np.min(area_ratios_log)) / (np.max(area_ratios_log) - np.min(area_ratios_log))

# Now to plot the heatmap
plt.figure(figsize=(10, 8))
trip = plt.tripcolor(
    points[:, 0], 
    points[:, 1], 
    tri.simplices, 
    facecolors=area_ratios_normalized, 
    edgecolors='none',  
    cmap='inferno',     
    shading='flat'    
)

plt.colorbar(trip, label='Area Ratio (3D / 2D)')
plt.title('Change in Triangle Area After Lift')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.savefig('heatmap_partbh.png', bbox_inches='tight')


# Part c

# Need the induced metric which is just a matrix
# If the metric is just a matrix called g, then I can write in each of the elements

# g_xx = 1 + (2x + y)^2
# g_yy = 1 + (x+2y)^2
# g_xy = (2x+y)(x+2y)
# g_yx = (2x+y)(x+2y)

# These were calculated by parametrizing the lift equation for z and taking partial derivatives

# Part d

def surface_norm(coords_3d):
    x = coords_3d[:, 0]
    y = coords_3d[:, 1]
    
    normals = np.zeros_like(coords_3d)
    normals[:, 0] = 2 * x + y
    normals[:, 1] = -(x + 2 * y)
    normals[:, 2] = 1
    
    norms = np.linalg.norm(normals, axis=1)
    return normals / norms[:, np.newaxis]

def mesh_plus_normals(coords_3d, normals, tri):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.7)
    
    ax.quiver(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], length=0.3, color='red', normalize=True)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lifted Mesh with Surface Normals')
    plt.show()
    plt.savefig('surface_normals_plus_mesh_parth.png', bbox_inches='tight')

normals = surface_norm(coords_3d)

# generate the plot

mesh_plus_normals(coords_3d, normals, tri)

# Part e
# Similar procedure to part d

from numpy import cross, pad, array, vstack

points3d = vstack((points.T, z(*points.T))).T

def tri_area(p1, p2, p3):
	p1 = pad(p1, (0, 3-len(p1)), 'constant')
	p2 = pad(p2, (0, 3-len(p2)), 'constant')
	p3 = pad(p3, (0, 3-len(p3)), 'constant')
	return ((cross(p2 - p1, p3 - p1)/2)**2).sum()**0.5

def triangle_normal(p1, p2, p3):
	normal = cross(p3 - p1, p2 - p1)
	return normal / (normal**2).sum()**0.5

vertex_normals = []
triangles = Delaunay(points).simplices
for i in range(len(points)):
	triangles_with_vertex = [t for t in triangles if i in t]
	normal = sum(triangle_normal(*points3d[i]) * tri_area(*points3d[i]) for i in triangles_with_vertex)
	vertex_normals.append(normal / (normal**2).sum()**0.5)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_trisurf(*points3d.T, triangles=triangles, cmap='viridis')
ax.quiver(*points3d.T, *array(vertex_normals).T, color='r', length=0.5)
plt.show()
plt.savefig('vertex_normals_plus_mesh_parth.png')

# Part f
# Need second fundamental form which (to my understanding) is a matrix for every point in the mesh
# This code will not produce a plotted output, but rather a matrix, similar to part c.

def compute_normal_vector(x, y):
    rx = np.array([1, 0, 2*x + y])
    ry = np.array([0, 1, x + 2*y])
    n = np.cross(rx, ry)
    norm = np.linalg.norm(n)
    return n / norm if norm != 0 else n

def compute_second_fundamental_form(x, y, normal):
    r_xx = np.array([0, 0, 2])
    r_xy = np.array([0, 0, 1]) 
    r_yy = np.array([0, 0, 2])
    L = np.dot(r_xx, normal)
    M = np.dot(r_xy, normal)
    N = np.dot(r_yy, normal)

    return np.array([[L, M], [M, N]])

second_forms = []
for point in points:
    x, y = point[0], point[1]
    normal = compute_normal_vector(x, y)
    II = compute_second_fundamental_form(x, y, normal)
    second_forms.append(II)

# Part g
# For the shape operator I need to employ the metric tensor and second fundamental form
# I only wrote the metric tensor for part c so I need to employ it as functioning code

def compute_metric_tensor(points):
    g_tensors = np.zeros((points.shape[0], 2, 2))
    
    for i, (x, y) in enumerate(points):
        g11 = 1 + (2*x + y)**2
        g22 = 1 + (x + 2*y)**2
        g12 = (2*x + y)*(x + 2*y)
        g_tensors[i] = [[g11, g12], [g12, g22]]
    
    return g_tensors

metric_tensors = compute_metric_tensor(points)

from mpl_toolkits.mplot3d import Axes3D
z = points[:, 0]**2 + points[:, 1]**2
points_3d = np.column_stack((points, z))  

shape_operators = np.array([np.linalg.inv(g).dot(II) for g, II in zip(metric_tensors, second_forms)])

gaussian_curvature = np.linalg.det(shape_operators)
mean_curvature = 0.5 * np.trace(shape_operators, axis1=1, axis2=2)

principal_curvatures = np.array([np.linalg.eigvals(S) for S in shape_operators])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Gaussian Curvature')
plt.title('Gaussian Curvature of the Surface')
plt.show()
plt.savefig('gaussian_parth.png')






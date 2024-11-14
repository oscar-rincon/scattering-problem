# utf-8
"""
Scattering of a plane wave by a rigid cylinder using the Mie series expansion.

This script calculates the displacement field for the scattering of a plane wave by a rigid cylinder
using the Mie series expansion. The displacement field is calculated as the sum of the incident and
scattered waves. The incident wave is a plane wave impinging on the cylinder, and the scattered wave
is the wave scattered by the cylinder. The displacement field is calculated in polar coordinates
(r, theta) and plotted in polar coordinates.

"""
import subprocess
import timeit
from scipy.special import jn, hankel2
import numpy as np
from numpy import pi, exp, cos, zeros_like, ma, real, round, min, max, std, mean
import matplotlib.pyplot as plt
import gmsh
import meshio
import matplotlib as mpl
from scipy.interpolate import griddata

 # Configuración de LaTeX para matplotlib
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "xelatex",        # change this if using xetex or lautex
    "text.usetex": False,                # use LaTeX to write all text
    "font.family": "sans-serif",
    # "font.serif": [],
    "font.sans-serif": ["DejaVu Sans"], # specify the sans-serif font
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # "figure.figsize": (3.15, 2.17),     # default fig size of 0.9 textwidth
    "pgf.preamble": r'\usepackage{amsmath},\usepackage{amsthm},\usepackage{amssymb},\usepackage{mathspec},\renewcommand{\familydefault}{\sfdefault},\usepackage[italic]{mathastext}'
    #"pgf.preamble": r' \usepackage{amsmath},\usepackage{cmbright},\usepackage[utf8x]{inputenc},\usepackage[T1]{fontenc},\usepackage{amssymb},\usepackage{amsfonts},\usepackage{mathastext}',
        # plots will be generated using this preamble
    }
mpl.rcParams.update(pgf_with_latex)

def u_exact_calc(r, theta, r_i, k, nmax=None):
    """
    Calculate the exact solution for the scattered and incident waves around a cylinder.

    Parameters:
    r (ndarray): Radial coordinates where the solution is evaluated.
    theta (ndarray): Angular coordinates where the solution is evaluated.
    r_i (float): Radius of the cylinder.
    k (float): Wave number.

    Returns:
    tuple: A tuple containing:
        - us_inc (ndarray): Incident wave field.
        - us_scn (ndarray): Scattered wave field.
        - u (ndarray): Total wave field (incident + scattered).
    """
    us_scn = zeros_like(r, dtype=complex)  # Initialize scattered wave
    us_inc = zeros_like(r, dtype=complex)  # Initialize scattered wave

    if nmax is None:
        nmax = int(30 + (k * r_i)**1.01)
    
    for n in range(nmax, -1, -1):
        if n == 0:
            # Coefficient for n = 0
            an = -jn(1, k*r_i) / hankel2(1, k*r_i)
            en = 1.0
        else:
            # Coefficients for n > 0
            an = -2.0 * (-1.0)**n * ((jn(n + 1, k*r_i) - jn(n - 1, k*r_i)) /
                                     (hankel2(n + 1, k*r_i) - hankel2(n - 1, k*r_i)))
            en = 2.0
        # Sum terms for both scattered and incident waves
        usn = an * 1.0j**n * hankel2(n, k*r) * cos(n*theta) * exp(1.0j*pi) 
        uin = en * 1.0j**n * jn(n, -k*r) * cos(n*theta) #* exp(1.0j*pi) 

        # Add terms to the total displacement field
        us_inc = us_inc + uin
        us_scn = us_scn + usn 
    
    # Total displacement field
    u = us_scn + us_inc

    # Extract the amplitude of the displacement
    u_scn_amp = np.real(us_scn)
    u_amp = np.real(u)   
    
    return u_scn_amp, u_amp 

def mask_displacement(R_exact, r_i, r_e, u_amp_exact, u_scn_amp_exact):
    """
    Mask the displacement outside the scatterer.

    Parameters:
    R_exact (numpy.ndarray): Radial coordinates.
    r_i (float): Inner radius.
    r_e (float): Outer radius.
    u_amp_exact (numpy.ma.core.MaskedArray): Exact displacement amplitude.
    u_scn_amp_exact (numpy.ma.core.MaskedArray): Exact scattered displacement amplitude.

    Returns:
    u_amp_exact (numpy.ma.core.MaskedArray): Masked exact displacement amplitude.
    u_scn_amp_exact (numpy.ma.core.MaskedArray): Masked exact scattered displacement amplitude.
    """
    u_amp_exact = np.ma.masked_where(R_exact < r_i, u_amp_exact)
    u_amp_exact = np.ma.masked_where(R_exact > r_e, u_amp_exact)
    u_scn_amp_exact = np.ma.masked_where(R_exact < r_i, u_scn_amp_exact)
    u_scn_amp_exact = np.ma.masked_where(R_exact > r_e, u_scn_amp_exact)
    return u_amp_exact, u_scn_amp_exact

def plot_displacement_amplitude(X, Y, u_scn_amp, u_amp):
    """
    Plot the amplitude of the scattered and total displacement.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_scn_amp (numpy.ma.core.MaskedArray): Amplitude of the scattered displacement.
    u_amp (numpy.ma.core.MaskedArray): Amplitude of the total displacement.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot u_scn_amp
    c1 = axs[0].pcolormesh(X, Y, u_scn_amp, cmap="RdYlBu")
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=0.7, orientation="horizontal", pad=0.07)
    cb1.set_label(r"Amplitude $u_{\rm{sct}}$")
    cb1.set_ticks([np.trunc(np.min(u_scn_amp) * 1e+2) / 1e+2, np.trunc(np.max(u_scn_amp) * 1e+2) / 1e+2])
    cb1.set_ticklabels([f'{(np.trunc(np.min(u_scn_amp) * 1e+2) / 1e+2):.2f}', f'{(np.trunc(np.max(u_scn_amp) * 1e+2) / 1e+2):.2f}'])
    
    axs[0].axis("off")
    axs[0].set_aspect("equal")

    # Plot u_amp
    c2 = axs[1].pcolormesh(X, Y, u_amp, cmap="RdYlBu")
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=0.7, orientation="horizontal", pad=0.07)
    cb2.set_label(r"Amplitude $u$")
    cb2.set_ticks([np.trunc(np.min(u_amp) * 1e+2) / 1e+2, np.trunc(np.max(u_amp) * 1e+2) / 1e+2])
    cb2.set_ticklabels([f'{(np.trunc(np.min(u_amp) * 1e+2) / 1e+2):.2f}', f'{(np.trunc(np.max(u_amp) * 1e+2) / 1e+2):.2f}'])
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    #plt.tight_layout()
    plt.show()
 


def plot_mesh_from_file(file_path_msh):
    """
    Reads a mesh file using meshio, extracts the points and cells, and plots the mesh.
    Also calculates the number of connections and nodes in the physical domain 
    and absorbing layer, and returns these values.

    Parameters:
    file_path_msh (str): Path to the mesh file.

    Returns:
    tuple: (num_nodes, num_connections_P, num_connections_A)
        - num_nodes: The total number of unique nodes in the mesh.
        - num_connections_P: The total number of unique connections in the physical domain.
        - num_connections_A: The total number of unique connections in the absorbing layer.
    """
    
    # Read the mesh using meshio
    mesh = meshio.read(file_path_msh)
    points = mesh.points 
    cells = mesh.cells

    # Extract the triangles from the mesh
    triangles_P = cells[10].data  # Connections for physical domain
    triangles_A = cells[9].data   # Connections for absorbing layer

    # Calculate the number of nodes (unique points)
    num_nodes = len(points)

    # Calculate the number of connections (unique edges)
    def count_unique_edges(triangles):
        edges = set()  # To store unique edges as tuples (sorted to avoid duplicates)
        for triangle in triangles:
            # Create all 3 edges of the triangle and sort each edge
            edges.add(tuple(sorted([triangle[0], triangle[1]])))
            edges.add(tuple(sorted([triangle[1], triangle[2]])))
            edges.add(tuple(sorted([triangle[2], triangle[0]])))
        return len(edges)

    # Count the number of unique edges in physical domain and absorbing layer
    num_connections_P = count_unique_edges(triangles_P)
    num_connections_A = count_unique_edges(triangles_A)

    # Print the results (optional)
    # print(f"Number of nodes: {num_nodes}")
    # print(f"Number of connections in physical domain: {num_connections_P}")
    # print(f"Number of connections in absorbing layer: {num_connections_A}")

    # Plot the mesh
    plt.figure(figsize=(4, 4))
    plt.triplot(points[:, 0], points[:, 1], triangles_P, color='#cbcbcbff', lw=0.3)
    plt.triplot(points[:, 0], points[:, 1], triangles_A, color='#989898ff', lw=0.3)

    # Surface connections (edges for physical and absorbing layers)
    connections_ri = np.concatenate([cells[i].data for i in range(0, 4)])   
    connections_re = np.concatenate([cells[i].data for i in range(4, 8)])   
    start_points_ri = points[connections_ri[:, 0]]
    end_points_ri = points[connections_ri[:, 1]]
    start_points_re = points[connections_re[:, 0]]
    end_points_re = points[connections_re[:, 1]]

    # Plot connections (edges)
    plt.plot(np.vstack([start_points_ri[:, 0], end_points_ri[:, 0]]), 
             np.vstack([start_points_ri[:, 1], end_points_ri[:, 1]]), color='#000146ff', lw=0.3)
    plt.plot(np.vstack([start_points_re[:, 0], end_points_re[:, 0]]), 
             np.vstack([start_points_re[:, 1], end_points_re[:, 1]]), color='#003e23ff', lw=0.3)

    # Final plot adjustments
    plt.axis('off')
    plt.show()

    # Return the calculated values
    return num_nodes, num_connections_P, num_connections_A    

def process_onelab_data(file_path):
    """
    Process GMSH data from a given file and extract node coordinates and displacement values.

    Parameters:
    file_path (str): The path to the GMSH file to be processed.

    Returns:
    tuple: A tuple containing three numpy arrays:
        - X (numpy.ndarray): The x-coordinates of the nodes.
        - Y (numpy.ndarray): The y-coordinates of the nodes.
        - u_amp (numpy.ndarray): The displacement values at the nodes.
    """
    # Verificar si GMSH ya está inicializado, si no lo está, inicializarlo
    gmsh.initialize()

    # Abrir el archivo de GMSH
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(file_path)

    # Extraer nodos y elementos
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()

    # Reshape de los elementos
    elements = np.array(element_node_tags[0]).reshape(-1, 3)

    # Reshape de las coordenadas de los nodos
    node_coords = node_coords.reshape(-1, 3)

    # Obtener los datos del campo
    try:
        field = gmsh.view.getHomogeneousModelData(0, 0)
    except:
        field = gmsh.view.getHomogeneousModelData(1, 0)
    field_id = field[1]
    field_data = np.array(field[2]).reshape(-1, 3)

    # Cerrar GMSH
    gmsh.finalize()

    # Crear el diccionario
    node_u_data_dict = {}
    for elem, u_vals in zip(elements, field_data):
        for node, u_val in zip(elem, u_vals):
            node_u_data_dict[node] = u_val

    # Crear el diccionario node_tags -> node_coords
    node_coords_dict = {}
    for tag, coord in zip(node_tags, node_coords):
        node_coords_dict[tag] = coord

    # Añadir los valores de desplazamiento a las coordenadas de los nodos
    for tag, coord in zip(node_tags, node_coords):
        node_coords_dict[tag][2] = node_u_data_dict[tag]

    # Crear un nuevo arreglo de nodos con los valores de desplazamiento
    pts = np.array([coord for coord in node_coords_dict.values()])

    # Create an array to store the reordered pts
    reordered_pts = np.zeros((int(np.max(node_tags)), 3))

    # Reorder pts based on the indices of node_tags
    for i, tag in enumerate(node_tags-1):
        reordered_pts[int(tag)] = pts[i]

    X = reordered_pts[:, 0]
    Y = reordered_pts[:, 1]
    u_amp = reordered_pts[:, 2]

    return X, Y, elements-1, u_amp

def plot_fem_results(X_fem, Y_fem, elements_fem, uscn_amp_fem, u_amp_fem):
    """
    Plots the FEM results for the scattered and total displacement amplitudes.

    Parameters:
    X_fem (numpy.ndarray): X coordinates of the FEM mesh.
    Y_fem (numpy.ndarray): Y coordinates of the FEM mesh.
    elements_fem (numpy.ndarray): Elements of the FEM mesh.
    uscn_amp_fem (numpy.ndarray): Scattered displacement amplitude from FEM.
    u_amp_fem (numpy.ndarray): Total displacement amplitude from FEM.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot the first result
    axs[0].tricontourf(X_fem, Y_fem, elements_fem, uscn_amp_fem, cmap="RdYlBu", levels=100)
    axs[0].triplot(X_fem, Y_fem, elements_fem, color='gray', lw=0.1)
    axs[0].axis('off')
    cbar1 = plt.colorbar(axs[0].collections[0], ax=axs[0], shrink=0.9, orientation="horizontal", pad=0.07)
    cbar1.set_ticks([uscn_amp_fem.min(), uscn_amp_fem.max()])
    cbar1.set_ticklabels([f'{uscn_amp_fem.min():.2f}', f'{uscn_amp_fem.max():.2f}'])
    cbar1.set_label(r"Amplitude $u_{\rm{sct}}$")
    axs[0].set_aspect('equal', adjustable='box')

    # Plot the second result
    axs[1].tricontourf(X_fem, Y_fem, elements_fem, u_amp_fem, cmap="RdYlBu", levels=100)
    axs[1].triplot(X_fem, Y_fem, elements_fem, color='gray', lw=0.1)
    axs[1].axis('off')
    cbar2 = plt.colorbar(axs[1].collections[0], ax=axs[1], shrink=0.9, orientation="horizontal", pad=0.07)
    cbar2.set_ticks([u_amp_fem.min(), u_amp_fem.max()])
    cbar2.set_ticklabels([f'{u_amp_fem.min():.2f}', f'{u_amp_fem.max():.2f}'])
    axs[1].set_aspect('equal', adjustable='box')
    cbar2.set_label(r"Amplitude $u$")
    plt.show()    

def interpolate_fem_data(X_fem, Y_fem, u_amp_fem, uscn_amp_fem, r_i, r_e, n_grid):
    """
    Interpolates FEM data onto a regular grid and masks the displacement outside the scatterer.

    Parameters:
    X_fem (numpy.ndarray): X coordinates of FEM data points.
    Y_fem (numpy.ndarray): Y coordinates of FEM data points.
    u_amp_fem (numpy.ndarray): Amplitude of the total displacement from FEM.
    uscn_amp_fem (numpy.ndarray): Amplitude of the scattered displacement from FEM.
    r_i (float): Inner radius of the scatterer.
    r_e (float): Outer radius of the scatterer.
    n_grid (int): Number of grid points for interpolation.

    Returns:
    X_grid (numpy.ndarray): X coordinates of the regular grid.
    Y_grid (numpy.ndarray): Y coordinates of the regular grid.
    u_amp_interp_fem (numpy.ma.core.MaskedArray): Interpolated and masked total displacement amplitude.
    uscn_amp_interp_fem (numpy.ma.core.MaskedArray): Interpolated and masked scattered displacement amplitude.
    """
    # Scale the data
    X_fem, Y_fem = X_fem*r_e, Y_fem*r_e

    # Create a regular grid where you want to compare the exact solution
    x_grid = np.linspace(min(X_fem), max(X_fem), n_grid)
    y_grid = np.linspace(min(Y_fem), max(Y_fem), n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate onto the grid without NaN values
    u_amp_interp_fem = griddata((X_fem, Y_fem), u_amp_fem, (X_grid, Y_grid), method='nearest')
    uscn_amp_interp_fem = griddata((X_fem, Y_fem), uscn_amp_fem, (X_grid, Y_grid), method='nearest')

    # Create R_grid and Theta_grid for the new grid
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    Theta_grid = np.arctan2(Y_grid, X_grid)

    # Mask the displacement outside the scatterer
    u_amp_interp_fem = np.ma.masked_where(R_grid < r_i, u_amp_interp_fem)
    u_amp_interp_fem = np.ma.masked_where(R_grid > r_e, u_amp_interp_fem)
    uscn_amp_interp_fem = np.ma.masked_where(R_grid < r_i, uscn_amp_interp_fem)
    uscn_amp_interp_fem = np.ma.masked_where(R_grid > r_e, uscn_amp_interp_fem)

    return X_grid, Y_grid, u_amp_interp_fem, uscn_amp_interp_fem

def calc_error(X, Y, u_scn_amp_exact, u_amp_exact, uscn_amp_interp, u_amp_interp, r_i, r_e):
    """
    Calculate the relative error between the exact and interpolated FEM solutions.

    Parameters:
    X (numpy.ndarray): X-coordinates of the grid.
    Y (numpy.ndarray): Y-coordinates of the grid.
    u_scn_amp_exact (numpy.ma.core.MaskedArray): Exact scattered displacement amplitude.
    u_amp_exact (numpy.ma.core.MaskedArray): Exact total displacement amplitude.
    uscn_amp_interp (numpy.ma.core.MaskedArray): Interpolated scattered displacement amplitude from FEM.
    u_amp_interp (numpy.ma.core.MaskedArray): Interpolated total displacement amplitude from FEM.
    R_exact (numpy.ndarray): Radial coordinates of the exact solution.
    r_i (float): Inner radius of the scatterer.
    r_e (float): Outer radius of the scatterer.

    Returns:
    tuple: Relative errors for scattered and total displacement amplitudes.
    """
    # Create R_grid for the new grid
    R_grid = np.sqrt(X**2 + Y**2)

    # Mask the displacement outside the scatterer
    u_scn_amp_exact_data = u_scn_amp_exact.data
    u_scn_amp_exact_data[(R_grid < r_i) | (R_grid > r_e)] = 0
    u_amp_exact_data = u_amp_exact.data
    u_amp_exact_data[(R_grid < r_i) | (R_grid > r_e)] = 0
    u_amp_interp_data = u_amp_interp.data
    u_amp_interp_data[(R_grid < r_i) | (R_grid > r_e)] = 0
    uscn_amp_interp_data = uscn_amp_interp.data
    uscn_amp_interp_data[(R_grid < r_i) | (R_grid > r_e)] = 0

    # Calculate the difference between the interpolated results and the exact results
    diff_uscn_amp_data, diff_u_amp_data = uscn_amp_interp_data - u_scn_amp_exact_data, u_amp_interp_data - u_amp_exact_data
    diff_uscn_amp, diff_u_amp = uscn_amp_interp - u_scn_amp_exact, u_amp_interp - u_amp_exact

    # Calculate the L2 norm of the differences
    norm_diff_uscn = np.linalg.norm(diff_uscn_amp_data, 2)
    norm_diff_u = np.linalg.norm(diff_u_amp_data, 2)

    # Calculate the L2 norm of the exact solutions
    norm_u_scn_exact = np.linalg.norm(u_scn_amp_exact_data, 2)
    norm_u_exact = np.linalg.norm(u_amp_exact_data, 2)

    # Calculate the relative errors using L2 norms
    rel_error_uscn = norm_diff_uscn / norm_u_scn_exact
    rel_error_u = norm_diff_u / norm_u_exact

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot u_scn_amp
    c1 = axs[0].pcolormesh(X, Y, diff_uscn_amp, cmap="RdYlBu", vmin=np.min(diff_uscn_amp), vmax=np.max(diff_uscn_amp))
    cb1 = fig.colorbar(c1, ax=axs[0], shrink=0.7, orientation="horizontal", pad=0.07)
    cb1.set_label(r"Error $u_{\rm{sct}}$")
    cb1.set_ticks([np.trunc(np.min(diff_uscn_amp) * 1e+2) / 1e+2, np.trunc(np.max(diff_uscn_amp) * 1e+2) / 1e+2])
    axs[0].axis("off")
    axs[0].set_aspect("equal")

    # Plot u_amp
    c2 = axs[1].pcolormesh(X, Y, diff_u_amp, cmap="RdYlBu", vmin=np.min(diff_u_amp), vmax=np.max(diff_u_amp))
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=0.7, orientation="horizontal", pad=0.07)
    cb2.set_label(r"Error $u$")
    cb2.set_ticks([np.trunc(np.min(diff_u_amp) * 1e+2) / 1e+2, np.trunc(np.max(diff_u_amp) * 1e+2) / 1e+2])
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    plt.show()

    return rel_error_uscn, rel_error_u

def measure_execution_time(getdp_path, command_args, num_runs=10):
    """
    Measures the execution time of a command run by the GetDP software.

    Parameters:
    getdp_path (str): The path to the GetDP executable.
    command_args (str): The command line arguments to pass to GetDP.
    num_runs (int, optional): The number of times to run the command for measurement. Default is 10.

    Returns:
    tuple: A tuple containing the average execution time, standard deviation of the execution time,
           minimum execution time, and maximum execution time.
    """
    def run_getdp():
        subprocess.run(f"{getdp_path} {command_args}", 
                       shell=True, 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)

    times = timeit.repeat(run_getdp, repeat=num_runs, number=1)
    average_time = mean(times)
    std_dev_time = std(times)
    min_time = min(times)
    max_time = max(times)

    return average_time, std_dev_time, min_time, max_time    
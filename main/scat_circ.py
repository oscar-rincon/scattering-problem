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

def u_exact_calc(r, theta, r_i, k):
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
    nmax = int(30 + (k * r_i)**1.01) # Number of terms in the series
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
        usn = an * 1.0j**n * hankel2(n, k*r) * cos(n*theta)
        uin = en * 1.0j**n * jn(n, -k*r) * cos(n*theta) * exp(1.0j*pi) 

        # Add terms to the total displacement field
        us_inc = us_inc + uin
        us_scn = us_scn + usn 
    
    # Total displacement field
    u = us_scn + us_inc

    # Extract the amplitude of the displacement
    u_scn_amp = np.real(us_scn)
    u_amp = np.real(u)   
     
    return u_scn_amp, u_amp 


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
    cb1.set_label("Amplitude $u_{\\text{sct}}$")
    cb1.set_ticks([np.round(np.min(u_scn_amp), 2), np.round(np.max(u_scn_amp), 2)])
    axs[0].axis("off")
    axs[0].set_aspect("equal")

    # Plot u_amp
    c2 = axs[1].pcolormesh(X, Y, u_amp, cmap="RdYlBu")
    cb2 = fig.colorbar(c2, ax=axs[1], shrink=0.7, orientation="horizontal", pad=0.07)
    cb2.set_label("Amplitude $u$")
    cb2.set_ticks([np.round(np.min(u_amp), 2), np.round(np.max(u_amp), 2)])
    axs[1].axis("off")
    axs[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()
 

def plot_mesh(file_path):
    """
    Plots a triangular mesh using matplotlib.

    Parameters:
    file_path (str): The path to the GMSH file to be processed.

    The function extracts the points and triangular cells from the mesh and 
    plots them using matplotlib's triplot function. The plot is displayed 
    without axis.
    """
    
    # Read the mesh using meshio
    mesh = meshio.read(file_path)
    points = mesh.points 
    cells = mesh.cells

    # Extract the triangles from the mesh
    triangles = cells[10].data

    # Plot the mesh
    plt.figure(figsize=(4, 4))
    plt.triplot(points[:, 0], points[:, 1], triangles, color='gray', lw=0.4)
    plt.axis('off')
    plt.show()


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


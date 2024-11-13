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
from numpy import pi, exp, cos, zeros_like, ma, real, round, min, max, std, mean
import matplotlib.pyplot as plt

def u_exact(r, theta, r_i, k):
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
    u = us_scn + us_inc    
    return us_inc, us_scn, u 


def plot_displacement_amplitude(R, Theta, r_i, u_inc, u_scn, u):
    """
    Plots the amplitude of the displacement in polar coordinates for the incident wave, 
    scattered wave, and total wave.

    Parameters:
    R (ndarray): Radial coordinates.
    Theta (ndarray): Angular coordinates.
    r_i (float): Inner radius to mask out regions where r < r_i.
    u_inc (ndarray): Incident wave displacement.
    u_scn (ndarray): Scattered wave displacement.
    u (ndarray): Total wave displacement.

    Returns:
    None
    """
    # Extract the amplitude of the displacement
    u_inc_amp = real(u_inc)
    u_scn_amp = real(u_scn)
    u_amp = real(u)

    # Mask out regions where r < 1 (inside the circle with radius 1)
    u_inc_amp = -ma.masked_where(R < r_i, u_inc_amp)
    u_scn_amp = -ma.masked_where(R < r_i, u_scn_amp)
    u_amp = -ma.masked_where(R < r_i, u_amp)

    # Plot the amplitude in polar coordinates
    plt.figure(figsize=(12, 4))

    # Plot the incident wave amplitude in polar coordinates
    ax3 = plt.subplot(1, 3, 1, projection='polar')
    c = ax3.pcolormesh(Theta, R, u_inc_amp, cmap="RdYlBu")
    cb = plt.colorbar(c, ax=ax3, shrink=0.8, orientation="horizontal", pad=0.07)
    cb.set_label("Amplitude $u_{\\text{inc}}$")
    ax3.set_xticklabels([])  # Remove radial labels
    ax3.set_yticklabels([])  # Remove radial labels
    ax3.grid(False)  # Hide grid lines
    cb.set_ticks([round(min(u_inc_amp), 2), round(max(u_inc_amp), 2)])

    # Plot the phase (angle of the displacement) in polar coordinates
    ax2 = plt.subplot(1, 3, 2, projection='polar')
    c = ax2.pcolormesh(Theta, R, real(u_scn_amp), cmap="RdYlBu")
    cb = plt.colorbar(c, ax=ax2, shrink=0.8, orientation="horizontal", pad=0.07)
    cb.set_label("Amplitude $u_{\\text{sct}}$")
    ax2.set_xticklabels([])  # Remove radial labels
    ax2.set_yticklabels([])  # Remove radial labels
    ax2.grid(False)  # Hide grid lines
    cb.set_ticks([round(min(u_scn_amp), 2), round(max(u_scn_amp), 2)])

    # Plot amplitude (real part of displacement) in polar coordinates
    ax1 = plt.subplot(1, 3, 3, projection='polar')
    c = ax1.pcolormesh(Theta, R, u_amp, cmap="RdYlBu")
    cb = plt.colorbar(c, ax=ax1, shrink=0.8, orientation="horizontal", pad=0.07)
    cb.set_label("Amplitude $u$")
    ax1.set_xticklabels([])  # Remove radial labels
    ax1.set_yticklabels([])  # Remove radial labels
    cb.set_ticks([round(min(u_amp), 2), round(max(u_amp), 2)])
    ax1.grid(False)  # Hide grid lines

    # Adjust the aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.tight_layout()
    plt.show()
 

def plot_mesh(mesh):
    """
    Plots a triangular mesh using matplotlib.

    Parameters:
    mesh (meshio.Mesh): A mesh object containing points and cells.

    The function extracts the points and triangular cells from the mesh and 
    plots them using matplotlib's triplot function. The plot is displayed 
    without axis.

    Example:
    >>> import meshio
    >>> mesh = meshio.read("path_to_mesh_file")
    >>> plot_mesh(mesh)
    """
    points = mesh.points 
    cells = mesh.cells
    # Extract the triangles from the mesh
    triangles = cells[10].data

    # Plot the mesh
    plt.figure(figsize=(5, 5))
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
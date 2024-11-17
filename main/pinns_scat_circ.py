import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import timeit

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_units, activation_function):
        """
        Initializes a more general neural network model.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
            hidden_layers (int): The number of hidden layers.
            hidden_units (int): The number of units in each hidden layer.
            activation_function (nn.Module): The activation function to use in the hidden layers.
        """
        super(MLP, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_units)
        self.linear_out = nn.Linear(hidden_units, output_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)])
        self.act = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x    

def set_seed(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility of results.

    Args:
        seed (int, optional): The seed value to use for random number generation. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Computes the derivative of a given tensor 'dy' with respect to another tensor 'x',
    up to a specified order.

    Args:
        dy (torch.Tensor): The tensor whose derivative is to be computed.
        x (torch.Tensor): The tensor with respect to which the derivative is to be computed.
        order (int, optional): The order of the derivative to compute. Defaults to 1, which
                               means a first-order derivative. Higher orders result in higher-order
                               derivatives.

    Returns:
        torch.Tensor: The computed derivative of 'dy' with respect to 'x', of the specified order.
    """
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True
        )[0]
    return dy  

def init_weights(m):
    """
    Initializes the weights and biases of a linear layer in the neural network using Xavier normalization.

    Args:
        m: The module or layer to initialize. If the module is of type nn.Linear, its weights and biases
           will be initialized.
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
      


def f(model, x_f, y_f, k):
    """
    Calculate the Helmholtz equation components for the given model and input domain.

    Parameters:
    model (torch.nn.Module): The neural network model.
    x_f (torch.Tensor): Tensor of x-coordinates of the input domain.
    y_f (torch.Tensor): Tensor of y-coordinates of the input domain.
    k (float): Wave number.

    Returns:
    torch.Tensor: Real part of the Helmholtz equation components.
    torch.Tensor: Imaginary part of the Helmholtz equation components.
    """
    # Concatenate x_f and y_f to form the input domain
    domain = torch.stack((x_f, y_f), dim=1)
    
    # Pass the domain through the model to get the output
    u = model(domain)
    
    # Extract real and imaginary parts
    u_real = u[:, 0]
    u_imag = u[:, 1]
    
    # Calculate second-order derivatives
    u_real_xx = derivative(u_real, x_f, order=2)
    u_real_yy = derivative(u_real, y_f, order=2)
    u_imag_xx = derivative(u_imag, x_f, order=2)
    u_imag_yy = derivative(u_imag, y_f, order=2)
    
    # Calculate the Helmholtz equation components
    f_u_real = u_real_xx + u_real_yy + k**2 * u_real
    f_u_imag = u_imag_xx + u_imag_yy + k**2 * u_imag
    
    return f_u_real, f_u_imag

def mse_f(model, x_f, y_f, k):
    """
    Calculate the mean squared error (MSE) for the Helmholtz equation components.

    Parameters:
    model (torch.nn.Module): The neural network model.
    x_f (torch.Tensor): Tensor of x-coordinates of the input domain.
    y_f (torch.Tensor): Tensor of y-coordinates of the input domain.
    k (float): Wave number.

    Returns:
    torch.Tensor: Mean squared error for the Helmholtz equation components.
    """
    # Calculate f(x, y) from the neural network
    f_u_real, f_u_imag = f(model, x_f, y_f, k)
    
    # Calculate the mean squared error for the real and imaginary parts
    error_f_real = torch.mean(f_u_real**2)
    error_f_imag = torch.mean(f_u_imag**2)
    
    # Sum the errors to obtain the total MSE
    mse = error_f_real + error_f_imag
      
    return mse

def mse_b(model, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k):
    """
    Calculate the mean squared error (MSE) for boundary conditions of a scattering problem.
    Parameters:
    model (torch.nn.Module): The neural network model used to approximate the solution.
    x_inner (torch.Tensor): x-coordinates of the inner boundary points.
    y_inner (torch.Tensor): y-coordinates of the inner boundary points.
    x_left (torch.Tensor): x-coordinates of the left boundary points.
    y_left (torch.Tensor): y-coordinates of the left boundary points.
    x_right (torch.Tensor): x-coordinates of the right boundary points.
    y_right (torch.Tensor): y-coordinates of the right boundary points.
    x_bottom (torch.Tensor): x-coordinates of the bottom boundary points.
    y_bottom (torch.Tensor): y-coordinates of the bottom boundary points.
    x_top (torch.Tensor): x-coordinates of the top boundary points.
    y_top (torch.Tensor): y-coordinates of the top boundary points.
    k (float): Wave number.
    Returns:
    float: The total mean squared error for all boundary conditions.
    """
    def calculate_mse_boundary(x, y, model, k, boundary_type):
        domain = torch.stack((x, y), axis=1)
        u = model(domain)
        u_real = u[:, 0]
        u_imag = u[:, 1]
        
        if boundary_type == 'inner':
            theta = torch.atan2(y, x)
            du_real_dx = derivative(u_real, x, order=1)
            du_real_dy = derivative(u_real, y, order=1)
            du_imag_dx = derivative(u_imag, x, order=1)
            du_imag_dy = derivative(u_imag, y, order=1)
            du_real_dn = -(torch.cos(theta) * du_real_dx + torch.sin(theta) * du_real_dy)
            du_imag_dn = -(torch.cos(theta) * du_imag_dx + torch.sin(theta) * du_imag_dy)
            ikx = 1j * k * x
            exp_ikx = 1j * k * torch.exp(ikx) * (torch.cos(theta))
            exp_ikx_real = torch.real(exp_ikx)
            exp_ikx_imag = torch.imag(exp_ikx)
            error_real = du_real_dn - exp_ikx_real
            error_imag = du_imag_dn - exp_ikx_imag
        
        elif boundary_type in ['left', 'right']:
            du_real_dx = derivative(u_real, x, order=1)
            du_imag_dx = derivative(u_imag, x, order=1)
            du_real_dn = (-1 if boundary_type == 'left' else 1) * du_real_dx
            du_imag_dn = (-1 if boundary_type == 'left' else 1) * du_imag_dx
            error_real = du_real_dn - (-k * u_imag)
            error_imag = du_imag_dn - (k * u_real)
        
        elif boundary_type in ['bottom', 'top']:
            du_real_dy = derivative(u_real, y, order=1)
            du_imag_dy = derivative(u_imag, y, order=1)
            du_real_dn = (-1 if boundary_type == 'bottom' else 1) * du_real_dy
            du_imag_dn = (-1 if boundary_type == 'bottom' else 1) * du_imag_dy
            error_real = du_real_dn - (-k * u_imag)
            error_imag = du_imag_dn - (k * u_real)
        
        mse = ((error_real)**2 + (error_imag)**2).mean()
        return mse

    mse_inner = calculate_mse_boundary(x_inner, y_inner, model, k, 'inner')
    mse_left = calculate_mse_boundary(x_left, y_left, model, k, 'left')
    mse_right = calculate_mse_boundary(x_right, y_right, model, k, 'right')
    mse_bottom = calculate_mse_boundary(x_bottom, y_bottom, model, k, 'bottom')
    mse_top = calculate_mse_boundary(x_top, y_top, model, k, 'top')

    mse = mse_inner + mse_left + mse_right + mse_bottom + mse_top
    return mse

def train_adam(model, x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k, num_iter=5_000):
 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global iter
     
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss_f = mse_f(model, x_f, y_f, k)
        loss_b = mse_b(model, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k)
        loss = loss_f + loss_b
        loss.backward(retain_graph=True)
        optimizer.step()
        iter += 1
        if iter % 1000 == 0:
            print(f"Adam - Iter: {iter} - Loss: {loss.item()}")


def closure(model, optimizer, x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k):
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Calculate the loss
    loss_f = mse_f(model, x_f, y_f, k)
    loss_b = mse_b(model, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k)
    loss = loss_b + loss_f
    
    # Backpropagate the loss
    loss.backward(retain_graph=True)
    
    # Update iteration counter and print loss every 100 iterations
    global iter
    iter += 1
    if iter % 1000 == 0:
        print(f"Iteration {iter}, Loss: {loss.item()}")
            
    return loss


# Function for L-BFGS training
def train_lbfgs(model, x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k, num_iter=5_000):

    optimizer = torch.optim.LBFGS(model.parameters(),
                                    lr=1,
                                    max_iter=num_iter,
                                    max_eval=num_iter,
                                    tolerance_grad=1e-7,
                                    history_size=100,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")
 
    closure_fn = partial(closure, model, optimizer, x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top, k)
    optimizer.step(closure_fn)


# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters
n_Omega_P = 5_000        # Number of points inside the annular region
n_Gamma_I = 100          # Number of points on the inner boundary (r = r_i)
n_Gamma_E = 200          # Number of points on the outer boundary (r = r_e)
r_i = np.pi / 4          # Inner radius
r_e = np.pi              # Outer radius
k = 3.0                  # Wave number
iter = 0                 # Iteration counter
side_length = 2 * r_e    # Side length of the square

def generate_points(n_Omega_P, side_length, r_i, n_Gamma_I, n_boundary_e):
    """
    Generate points inside the domain and on the boundaries.

    Parameters:
    n_Omega_P (int): Number of points inside the annular region.
    side_length (float): Side length of the square.
    r_i (float): Inner radius.
    n_Gamma_I (int): Number of points on the inner boundary.
    n_boundary_e (int): Number of points on each boundary.
    device (torch.device): Device to store the tensors.

    Returns:
    tuple: Tensors of x and y coordinates for points inside the domain, 
           on the inner boundary, and on the left, right, bottom, and top boundaries.
    """
    # Set the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Generate random samples for points inside the square but outside the circle
    points = []
    while len(points) < n_Omega_P:
        x_samples = side_length * (np.random.rand(n_Omega_P) - 0.5)
        y_samples = side_length * (np.random.rand(n_Omega_P) - 0.5)
        mask = (x_samples**2 + y_samples**2) >= r_i**2
        points.extend(zip(x_samples[mask], y_samples[mask]))

    # Trim the list to the desired number of points
    points = points[:n_Omega_P]
    x_samples, y_samples = zip(*points)

    # Convert to numpy arrays
    x_f = np.array(x_samples)
    y_f = np.array(y_samples)

    # Generate random points on the inner boundary (r = r_i)
    theta_inner = 2 * np.pi * np.random.rand(n_Gamma_I)  # Uniform angular distribution
    x_inner = r_i * np.cos(theta_inner)
    y_inner = r_i * np.sin(theta_inner)

    # Generate random points on the left, right, bottom, and top boundaries of the square

    # Left boundary (excluding corners)
    y_left = side_length * (np.random.rand(n_boundary_e) - 0.5)
    x_left = -side_length / 2 * np.ones_like(y_left)

    # Right boundary (excluding corners)
    y_right = side_length * (np.random.rand(n_boundary_e) - 0.5)
    x_right = side_length / 2 * np.ones_like(y_right)

    # Bottom boundary (excluding corners)
    x_bottom = side_length * (np.random.rand(n_boundary_e) - 0.5)
    y_bottom = -side_length / 2 * np.ones_like(x_bottom)

    # Top boundary (excluding corners)
    x_top = side_length * (np.random.rand(n_boundary_e) - 0.5)
    y_top = side_length / 2 * np.ones_like(x_top)

    # Convert to torch tensors
    x_f = torch.from_numpy(x_f).float().to(device).requires_grad_(True)
    y_f = torch.from_numpy(y_f).float().to(device).requires_grad_(True)
    x_inner = torch.from_numpy(x_inner).float().to(device).requires_grad_(True)
    y_inner = torch.from_numpy(y_inner).float().to(device).requires_grad_(True)
    x_left = torch.from_numpy(x_left).float().to(device).requires_grad_(True)
    y_left = torch.from_numpy(y_left).float().to(device).requires_grad_(True)
    x_right = torch.from_numpy(x_right).float().to(device).requires_grad_(True)
    y_right = torch.from_numpy(y_right).float().to(device).requires_grad_(True)
    x_bottom = torch.from_numpy(x_bottom).float().to(device).requires_grad_(True)
    y_bottom = torch.from_numpy(y_bottom).float().to(device).requires_grad_(True)
    x_top = torch.from_numpy(x_top).float().to(device).requires_grad_(True)
    y_top = torch.from_numpy(y_top).float().to(device).requires_grad_(True)

    return x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top
 
def plot_points(x_f, y_f, x_inner, y_inner, x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top):
    """
    Plot the points in the domain and on the boundaries.

    Parameters:
    x_f (torch.Tensor): x-coordinates of points inside the domain.
    y_f (torch.Tensor): y-coordinates of points inside the domain.
    x_inner (torch.Tensor): x-coordinates of points on the inner boundary.
    y_inner (torch.Tensor): y-coordinates of points on the inner boundary.
    x_left (torch.Tensor): x-coordinates of points on the left boundary.
    y_left (torch.Tensor): y-coordinates of points on the left boundary.
    x_right (torch.Tensor): x-coordinates of points on the right boundary.
    y_right (torch.Tensor): y-coordinates of points on the right boundary.
    x_bottom (torch.Tensor): x-coordinates of points on the bottom boundary.
    y_bottom (torch.Tensor): y-coordinates of points on the bottom boundary.
    x_top (torch.Tensor): x-coordinates of points on the top boundary.
    y_top (torch.Tensor): y-coordinates of points on the top boundary.
    """
    plt.figure(figsize=(4, 4))
    plt.scatter(x_f.cpu().detach().numpy(), y_f.cpu().detach().numpy(), c='#989898ff', s=2, marker='.', label=r"$\bf{x}$ $\in$ $\Omega_{\rm P}$")
    plt.scatter(x_inner.cpu().detach().numpy(), y_inner.cpu().detach().numpy(), c='#0000ffff', s=2, marker='.', label=r"$\bf{x}$ $\in$ $\Gamma_{\rm I}$")
    plt.scatter(x_left.cpu().detach().numpy(), y_left.cpu().detach().numpy(), c='#008000ff', s=2, marker='.', label=r"$\bf{x}$ $\in$ $\Gamma_{\rm E}$")
    plt.scatter(x_right.cpu().detach().numpy(), y_right.cpu().detach().numpy(), c='#008000ff', s=2, marker='.')
    plt.scatter(x_bottom.cpu().detach().numpy(), y_bottom.cpu().detach().numpy(), c='#008000ff', s=2, marker='.')
    plt.scatter(x_top.cpu().detach().numpy(), y_top.cpu().detach().numpy(), c='#008000ff', s=2, marker='.')
    plt.gca().set_aspect('equal', adjustable='box')

    # Set the ticks to include -pi and pi
    plt.xticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
    plt.yticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])

    # Adjust the legend position and remove the box
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), frameon=False, ncol=3)

    # Show the plot
    plt.show()


def initialize_and_load_model(model_path):
    """
    Initializes an MLP model and loads pre-trained weights from the specified path.
    Args:
        model_path (str): The file path to the pre-trained model weights.
    Returns:
        torch.nn.Module: The initialized MLP model with loaded weights.
    The function performs the following steps:
    1. Sets the device to 'cuda' if a GPU is available, otherwise 'cpu'.
    2. Initializes an MLP model with the specified architecture:
       - Input size: 2
       - Output size: 2
       - Hidden layers: 3
       - Hidden units per layer: 350
       - Activation function: Tanh
    3. Loads the pre-trained model weights from the given model_path.
    4. Sets the model to evaluation mode.
    """

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = MLP(input_size=2, output_size=2, hidden_layers=3, hidden_units=350, activation_function=nn.Tanh()).to(device)
    
    # Load the pre-trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def predict_u(model, r_e, r_i, k, dom_samples=500):
    """
    Calculate the real part of the scattered field for a given model.

    Parameters:
    model (torch.nn.Module): The neural network model.
    r_e (float): Outer radius.
    r_i (float): Inner radius.
    k (float): Wave number.
    dom_samples (int): Number of samples in the domain.

    Returns:
    numpy.ma.core.MaskedArray: The masked scattered field.
    numpy.ma.core.MaskedArray: The total field.
    """
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # x and y coordinates
    x = np.linspace(-r_e, r_e, dom_samples)
    y = np.linspace(-r_e, r_e, dom_samples)

    # Meshgrid of the domain
    X, Y = np.meshgrid(x, y)

    R_exact = np.sqrt(X**2 + Y**2)

    # Convert X and Y data to PyTorch tensors and reshape
    X_ten = torch.tensor(X).float().reshape(-1, 1).to(device)
    Y_ten = torch.tensor(Y).float().reshape(-1, 1).to(device)

    # Concatenate X and Y tensors into a single tensor
    domain_ten = torch.cat([X_ten, Y_ten], dim=1)
    u_sc_pred = model(domain_ten)
    u_sc_pred = u_sc_pred[:, 0].detach().cpu().numpy().reshape(X.shape)

    u_sc_pred = np.ma.masked_where(R_exact < r_i, u_sc_pred)

    us_inc = np.exp(1j * k * X)
    u_pred = np.real(us_inc + u_sc_pred)
    return u_sc_pred, u_pred

def measure_model_time_pinns(model, r_e, r_i, k, n_grid, num_runs=10):
    """
    Measure the time required to use the model.

    Parameters:
    model (torch.nn.Module): The neural network model.
    r_e (float): Outer radius.
    r_i (float): Inner radius.
    k (float): Wave number.
    n_grid (int): Number of grid points.
    num_runs (int): Number of runs to measure the time.

    Returns:
    dict: A dictionary containing average time, standard deviation, minimum time, and maximum time.
    """
    times = timeit.repeat(lambda: predict_u(model, r_e, r_i, k, n_grid), repeat=num_runs, number=1)
    return {
        'average_time': np.mean(times),
        'std_dev_time': np.std(times),
        'min_time': min(times),
        'max_time': max(times)
    }
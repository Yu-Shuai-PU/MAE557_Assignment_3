import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physical Constants and Non-Dimensional Parameters ---

Ma = 0.05
file_path = f'output_Ma_{Ma}.txt'
Re = 100
p0 = 101300.0
T0 = 300.0
R = 287.0
gamma = 1.4
nu = 0.0868
thermal_lambda = 0.1240
rho0 = p0 / (R * T0) # Reference density

L = 0.025 / Ma # 0.125 * 2 / 7 for Ma = 0.7; 0.125/3 = 0.041667 for Ma = 0.6; 0.0625 for Ma = 0.4, 0.125 for Ma = 0.2; 0.25 for Ma = 0.1; 0.5 for Ma = 0.05
U_wall = Ma * np.sqrt(gamma * R * T0) # Reference velocity
omega = 2 * nu / L**2

# --- Calculate Non-Dimensional Scales ---
t_c = L / U_wall   # Characteristic time
p_c = rho0 * U_wall**2 # Characteristic pressure (Dynamic pressure scale)
T_c = T0           # Characteristic temperature
L_c = L            # Characteristic length
u_c = U_wall       # Characteristic velocity
rho_c = rho0       # Characteristic density

print("--- Non-Dimensional Scales ---")
print(f"L_c (L): {L_c:.4e} m")
print(f"u_c (U_wall): {u_c:.4e} m/s")
print(f"rho_c (rho0): {rho_c:.4e} kg/m^3")
print(f"p_c (rho0 * U_wall^2): {p_c:.4e} Pa")
print(f"t_c (L/U_wall): {t_c:.4e} s")
print("--------------------------------")

# --- 2. Load *All* Data Columns ---

data = []
try:
    with open(file_path, 'r') as f:
        next(f) # Skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) == 8:
                # x, y, rho, u, v, p, T, Ma_local
                data.append([float(p) for p in parts])
            else:
                print(f"Warning: Skipping malformed line: {line}")

except FileNotFoundError:
    print(f"Error: File not found '{file_path}'")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

if not data:
    print("Error: No data was read from the file.")
    exit()

# Convert list to NumPy array
data_array = np.array(data)
print(f"Successfully loaded {len(data_array)} data points.")

# --- 3. Non-Dimensionalize *All* Variables ---

x_flat = data_array[:, 0] / L_c
y_flat = data_array[:, 1] / L_c
rho_flat = data_array[:, 2] / rho_c
u_flat = data_array[:, 3] / u_c
v_flat = data_array[:, 4] / u_c
p_flat = data_array[:, 5] / p_c
T_flat = data_array[:, 6] / T_c 
# Note: Pressure can be non-dimensionalized in various ways.
# p / p_c (using dynamic pressure) is used here.

# --- 4. Determine Grid and Reshape Data ---

x_unique = np.unique(x_flat)
y_unique = np.unique(y_flat)
Nx = len(x_unique)
Ny = len(y_unique)

print(f"Detected non-dimensional grid size: Nx = {Nx}, Ny = {Ny}")

if Nx * Ny != len(x_flat):
    print(f"Warning: Total data points ({len(x_flat)}) do not match Nx * Ny ({Nx * Ny}).")
    exit()

# Create the non-dimensional meshgrid
X, Y = np.meshgrid(x_unique, y_unique)

try:
    # Reshape *all* non-dimensional variables
    U = u_flat.reshape(Ny, Nx)
    V = v_flat.reshape(Ny, Nx)
    Rho = rho_flat.reshape(Ny, Nx)
    P = p_flat.reshape(Ny, Nx)
    T = T_flat.reshape(Ny, Nx)
except ValueError as e:
    print(f"Error reshaping data: {e}")
    exit()

print("Data reshaping complete.")

# --- 5. Calculate Non-Dimensional Key Physics ---

print("Calculating non-dimensional derivatives...")

# Calculate gradients. Since X, Y are non-dim,
# np.gradient automatically computes d/d(x/L) and d/d(y/L)
(dU_dy, dU_dx) = np.gradient(U, y_unique, x_unique)
(dV_dy, dV_dx) = np.gradient(V, y_unique, x_unique)

# Non-dimensional Divergence: div*(V*) = (dU*/dx*) + (dV*/dy*)
Divergence = dU_dx + dV_dy

# Non-dimensional Vorticity: omega* = (dV*/dx*) - (dU*/dy*)
Vorticity = dV_dx - dU_dy

# Non-dimensional Density gradient: |grad*(rho*)|
(dRho_dy, dRho_dx) = np.gradient(Rho, y_unique, x_unique)
Rho_gradient = np.sqrt(dRho_dx**2 + dRho_dy**2)

# Non-dimensional Velocity Magnitude
Magnitude = np.sqrt(U**2 + V**2)


# --- 6. Modular Plotting Functions ---

def plot_streamlines(X, Y, U, V, Mag):
    """Plots the non-dimensional streamlines."""
    print("Plotting streamlines...")
    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.pcolormesh(X, Y, Mag, shading='auto', cmap='viridis', alpha=0.7, vmin = 0, vmax = 0.6)
    fig.colorbar(c, ax=ax, label='Velocity Magnitude (U / U_wall)', shrink=0.8) 
    ax.streamplot(X, Y, U, V, density=10, color='black', linewidth=0.4)

    ax.set_title(f'Streamline Plot (Ma = {Ma})')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')

    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_vorticity(X, Y, Vorticity):
    """Plots the non-dimensional vorticity field (OO-style)."""
    print("Plotting vorticity...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # v_max = np.percentile(np.abs(Vorticity), 99.5) # Clip color range
    
    c = ax.pcolormesh(X, Y, Vorticity, shading='auto', cmap='RdBu_r', 
                       vmin=-10, vmax=10)
    
    fig.colorbar(c, ax=ax, label='Vorticity (omega * L / U_wall)', 
                 shrink=0.8, ticks = [-10, -5, 0, 5, 10]) # 'shrink' makes colorbar fit nicely
    
    ax.set_title(f'Vorticity Plot (Ma = {Ma})')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')
    
    # --- The Key Fix ---
    ax.set_aspect('equal', adjustable='box')
    # ---------------------
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_divergence(X, Y, Divergence):
    """Plots the non-dimensional divergence field (OO-style)."""
    print("Plotting divergence...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # d_max = np.percentile(np.abs(Divergence), 99.5) 
    
    c = ax.pcolormesh(X, Y, Divergence, shading='auto', cmap='RdBu_r', 
                       vmin=-0.5, vmax=0.5)
    
    fig.colorbar(c, ax=ax, label='Divergence * L / U_wall', 
                 shrink=0.8, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    
    ax.set_title(f'Divergence Plot (Ma = {Ma})')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')

    # --- The Key Fix ---
    ax.set_aspect('equal', adjustable='box')
    # ---------------------
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
    
    # These print statements are fine as they are
    print(f"Non-dim divergence range: {Divergence.min():.2e} to {Divergence.max():.2e}")
    print(f"(For reference, Ma^2 = {Ma**2:.2e})")

def plot_density(X, Y, Rho):
    """Plots the non-dimensional density field (OO-style)."""
    print("Plotting density...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    c = ax.pcolormesh(X, Y, Rho, shading='auto', cmap='RdBu_r', vmin = 0.5, vmax = 1.5)
    
    fig.colorbar(c, ax=ax, label='Density (rho / rho_0)', shrink=0.8, ticks=[0.5, 1.0, 1.5])
    
    ax.set_title(f'Density Plot (Ma = {Ma})')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')

    # --- The Key Fix ---
    ax.set_aspect('equal', adjustable='box')
    # ---------------------
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_density_gradient(X, Y, Rho_gradient):
    """Plots the non-dimensional Density gradient (OO-style)."""
    print("Plotting numerical density gradient...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # s_max = np.percentile(Rho_gradient, 99.5) # Clip for better contrast
    
    c = ax.pcolormesh(X, Y, Rho_gradient, shading='auto', cmap='Reds', 
                       vmin=0, vmax=3)

    fig.colorbar(c, ax=ax, label='|grad*(rho*)|', shrink=0.8, ticks=[0, 1, 2, 3])

    ax.set_title(f'Numerical Density Gradient Plot (Ma = {Ma})')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')
    
    # --- The Key Fix ---
    ax.set_aspect('equal', adjustable='box')
    # ---------------------
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()    
    plt.show()
    
def plot_temperature(X, Y, T):
    """Plots the non-dimensional temperature (OO-style)."""
    print("Plotting temperature...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    c = ax.pcolormesh(X, Y, T, shading='auto', cmap='inferno', vmin = 1.00, vmax = 1.02) 
                                                
    fig.colorbar(c, ax=ax, label='Temperature (T / T_0)', shrink=0.8, ticks=[1.00, 1.005, 1.01, 1.015, 1.02])
    
    ax.set_title(f'Temperature Plot (Ma = {Ma})')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')

    # --- The Key Fix ---
    ax.set_aspect('equal', adjustable='box')
    # ---------------------
    
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    plt.tight_layout()    
    plt.show()

# --- 7. Execute Plotting ---

# 1. Streamlines (Your original plot, now non-dim)
plot_streamlines(X, Y, U, V, Magnitude)

# 4. Vorticity (As you suggested)
plot_vorticity(X, Y, Vorticity)

# 2. Divergence (The definitive proof of compressibility)
# For Ma=0.4, this will be small but non-zero.
plot_divergence(X, Y, Divergence)

# 3. Density gradient (To "see" compression/expansion waves)
# For Ma=0.4, you should see faint gradients at the shear layer.
plot_density_gradient(X, Y, Rho_gradient)

# 5. Density
plot_density(X, Y, Rho)

# 6. Temperature
plot_temperature(X, Y, T)
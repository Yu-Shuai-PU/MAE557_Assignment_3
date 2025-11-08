import numpy as np
import matplotlib.pyplot as plt
import os, glob, re

# --------------------
# Parameters
# --------------------
Ma = 0.025
N = 64
dt = 0.1
t_final = 500
sample_interval = int(5.0 / dt)


OUTPUT_DIR = f"output_Ma_{Ma}_N_{N}_dt_{dt}_tfinal_{t_final}_sample_{sample_interval}/"
OUTPUT_DIR_U1  = os.path.join(OUTPUT_DIR, "u")
OUTPUT_DIR_U2  = os.path.join(OUTPUT_DIR, "v")
OUTPUT_DIR_p   = os.path.join(OUTPUT_DIR, "p")

def _get_filenumber(path):
    """Get file number from filename like 'rho_100.txt'."""
    m = re.search(r'_(\d+)\.txt$', os.path.basename(path))
    return int(m.group(1)) if m else -1

def _pick_frame_paths(frame='last'):
    """Return the four file paths for the specified frame (rho/u/v/T). frame can be 'last' or a specific integer."""
    u_files  = sorted(glob.glob(os.path.join(OUTPUT_DIR_U1 , "u_*.txt" )), key=_get_filenumber)
    v_files  = sorted(glob.glob(os.path.join(OUTPUT_DIR_U2 , "v_*.txt" )), key=_get_filenumber)
    p_files   = sorted(glob.glob(os.path.join(OUTPUT_DIR_p  , "p_*.txt"  )), key=_get_filenumber)

    if not (u_files and v_files and p_files):
        raise FileNotFoundError("No data files found in one or more directories.")

    # Decide which frame to use
    if frame == 'last':
        idx = min(len(u_files), len(v_files), len(p_files)) - 1
    else:
        # Find the file that matches the frame in the file number space (more robust, avoids sparse numbering)
        def _find_by_num(files, num):
            for p in files:
                if _get_filenumber(p) == num:
                    return p
            return None

        # If any file is missing, raise error
        paths = []
        for files, tag in [(u_files,'u'), (v_files,'v'), (p_files,'p')]:
            p = _find_by_num(files, frame)
            if p is None:
                raise FileNotFoundError(f"Can't find {tag}_{frame}.txt")
            paths.append(p)
        return tuple(paths)

    # Return the paths for the selected index
    return u_files[idx], v_files[idx], p_files[idx]

def plot_snapshot(frame='last', save_path=None, dpi=180):
    """
    Plot a snapshot of the simulation at a specific frame.
    Parameters:
      - frame: 'last' or a specific integer (e.g., 100)
      - save_path: If given, save as this PNG file; otherwise, display directly
    """
    # Get file paths
    u_path, v_path, p_path = _pick_frame_paths(frame)

    # Read data
    u  = np.loadtxt(u_path)
    v  = np.loadtxt(v_path)
    p   = np.loadtxt(p_path)
    speed = np.sqrt(u**2 + v**2)

    # Draw the meshgrid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    """Plots the non-dimensional streamlines."""
    print("Plotting streamlines...")
    fig, ax = plt.subplots(figsize=(7, 6))
    c = ax.pcolormesh(X, Y, speed, shading='auto', cmap='viridis', alpha=0.7, vmin=0, vmax=0.6)
    fig.colorbar(c, ax=ax, label='Velocity Magnitude (U / U_wall)', shrink=0.8)
    ax.streamplot(X, Y, u, v, density=10, color='black', linewidth=0.4)

    ax.set_title(f'Streamline Plot (Incompressible)')
    ax.set_xlabel('x / L')
    ax.set_ylabel('y / L')

    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    plot_snapshot(frame='last')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob
import sys
import re

# ==============================================================================
# --- Simulation Parameters ---
# ==============================================================================
Ma = 0.8
NX1 = 9
NX2 = 9
T_FINAL = 500
dt = 0.01
sample_interval = 500
dt_sample = dt * sample_interval
NUM_OUTPUT_FILES = 1 + int(T_FINAL / dt_sample)

# ==============================================================================
# --- Script Configuration ---
# ==============================================================================
OUTPUT_DIR = f"output_Ma_{Ma}_Nx_{NX1}_dt_{dt}/"
OUTPUT_DIR_RHO = os.path.join(OUTPUT_DIR, "rho")
OUTPUT_DIR_U1 = os.path.join(OUTPUT_DIR, "u1")
OUTPUT_DIR_U2 = os.path.join(OUTPUT_DIR, "u2")
OUTPUT_DIR_T = os.path.join(OUTPUT_DIR, "T")
ANIMATION_FILENAME = f"lid_driven_cavity_Ma_{Ma}_Nx_{NX1}_dt_{dt}.gif"
FPS = 5 # (Frames Per Second)


def animate_flow_field():
    """
    Scans the output directory, loads the simulation data, and creates
    an animation of the flow field evolution.
    """
    print("Starting visualization process...")

    # Get the filenumber from filename for proper sorting
    def get_filenumber(filepath):
        
        filename = os.path.basename(filepath)
        match = re.search(r'_(\d+).txt', filename)
        if match:
            # Extract and return the number as an integer
            return int(match.group(1))
        else:
            # If no match is found, return -1 to place it first
            return -1

    # Sort files numerically based on the filenumber
    try:
        rho_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_RHO, "rho_*.txt")), key=get_filenumber)
        u1_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_U1, "u1_*.txt")), key=get_filenumber)
        u2_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_U2, "u2_*.txt")), key=get_filenumber)
        T_files = sorted(glob.glob(os.path.join(OUTPUT_DIR_T, "T_*.txt")), key=get_filenumber)
        
        if not rho_files:
            print(f"Error: No data files found in '{OUTPUT_DIR}' directory.")
            print("Please run the C++ simulation first.")
            sys.exit(1)
            
        num_frames = len(rho_files)
        print(f"Found {num_frames} data snapshots to animate.")

    except Exception as e:
        print(f"Error while scanning for files: {e}")
        sys.exit(1)

    # --- 3. Pre-scan all files to determine global ranges for the colorbars ---
    print("Pre-scanning data to find global colorbar limits...")
    rho_min_global, rho_max_global = np.inf, -np.inf
    T_min_global, T_max_global = np.inf, -np.inf
    speed_min_global, speed_max_global = np.inf, -np.inf

    for i in range(num_frames):
        try:
            rho = np.loadtxt(rho_files[i])
            u1 = np.loadtxt(u1_files[i])
            u2 = np.loadtxt(u2_files[i])
            T = np.loadtxt(T_files[i])
            speed = np.sqrt(u1**2 + u2**2)
            
            if rho.min() < rho_min_global: rho_min_global = rho.min()
            if T.min() < T_min_global: T_min_global = T.min()
            if speed.min() < speed_min_global: speed_min_global = speed.min()
            
            if rho.max() > rho_max_global: rho_max_global = rho.max()
            if T.max() > T_max_global: T_max_global = T.max()
            if speed.max() > speed_max_global: speed_max_global = speed.max()
        except Exception as e:
            print(f"Warning: Skipping corrupted file {rho_files[i]} during pre-scan.")
            
    print(f"Global Rho range: [{rho_min_global:.3e}, {rho_max_global:.3e}]")
    print(f"Global T range: [{T_min_global:.3e}, {T_max_global:.3e}]")
    print(f"Global Speed range: [{speed_min_global:.3e}, {speed_max_global:.3e}]")


    # --- 4. Setup the plot grid ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    time_text = fig.suptitle("", fontsize=16) 

    x = np.linspace(0, 1, NX1)
    y = np.linspace(0, 1, NX2)
    X, Y = np.meshgrid(x, y)
    
    time_per_snapshot = T_FINAL / (num_frames - 1) if num_frames > 1 else 0

    rho_0 = np.loadtxt(rho_files[0])
    u1_0 = np.loadtxt(u1_files[0])
    u2_0 = np.loadtxt(u2_files[0])
    T_0 = np.loadtxt(T_files[0])
    speed_0 = np.sqrt(u1_0**2 + u2_0**2)

    im_rho = axes[0, 0].imshow(rho_0, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', animated=True)
    im_T = axes[0, 1].imshow(T_0, extent=[0, 1, 0, 1], origin='lower', cmap='inferno', animated=True)
    im_speed = axes[1, 0].imshow(speed_0, extent=[0, 1, 0, 1], origin='lower', cmap='plasma', animated=True)
    
    axes[1, 1].streamplot(X, Y, u1_0, u2_0, color='black', density=1.5, linewidth=0.8)

    # --- 5. Set fixed colorbar ranges ---
    im_rho.set_clim(vmin=rho_min_global, vmax=rho_max_global)
    im_T.set_clim(vmin=T_min_global, vmax=T_max_global)
    im_speed.set_clim(vmin=speed_min_global, vmax=speed_max_global)
    
    cbar_rho = fig.colorbar(im_rho, ax=axes[0, 0], label='$\\rho^*$')
    cbar_T = fig.colorbar(im_T, ax=axes[0, 1], label='$T^*$')
    cbar_speed = fig.colorbar(im_speed, ax=axes[1, 0], label='$|\\mathbf{u}^*|$')
    
    axes[0, 0].set_title("Density Field")
    axes[0, 1].set_title("Temperature Field")
    axes[1, 0].set_title("Speed Magnitude")
    axes[1, 1].set_title("Velocity Streamlines")
    for ax in axes.flatten():
        ax.set_xlabel('$x_1^*$')
        ax.set_ylabel('$x_2^*$')
        ax.set_aspect('equal')

    # --- 6. Define the animation update function ---
    def update(frame_index):
        current_time = frame_index * time_per_snapshot
        print(f"Processing {frame_index+1}/{num_frames} frames (t* = {current_time:.3f})")

        try:
            rho = np.loadtxt(rho_files[frame_index])
            u1 = np.loadtxt(u1_files[frame_index])
            u2 = np.loadtxt(u2_files[frame_index])
            T = np.loadtxt(T_files[frame_index])
            speed = np.sqrt(u1**2 + u2**2)
        except Exception as e:
            print(f"\nError loading data for frame {frame_index}, skipping.")
            return []
        
        im_rho.set_data(rho)
        im_T.set_data(T)
        im_speed.set_data(speed)

        ax_stream = axes[1, 1]
        ax_stream.clear() 
        ax_stream.streamplot(X, Y, u1, u2, color='black', density=1.5, linewidth=0.8)
        ax_stream.set_title("Velocity Streamlines")
        ax_stream.set_xlabel('$x_1^*$')
        ax_stream.set_ylabel('$x_2^*$')
        ax_stream.set_xlim([0, 1])
        ax_stream.set_ylim([0, 1])
        ax_stream.set_aspect('equal')

        time_text.set_text(f"2D Compressible Lid-Driven Cavity Flow (t* = {current_time:.3f})")
        
        return [im_rho, im_T, im_speed, time_text]

    # --- 7. Create and save the animation ---
    print("\nCreating animation... This may take a few minutes.")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000/FPS, blit=False)

    try:
        ani.save(ANIMATION_FILENAME, writer='pillow', fps=FPS)
        print(f"\nAnimation successfully saved as '{ANIMATION_FILENAME}'")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Please ensure you have 'pillow' installed (`pip install pillow`)")
        print("You might also need to install ffmpeg for saving as .mp4.")

if __name__ == '__main__':
    animate_flow_field()
import numpy as np
import matplotlib.pyplot as plt

def plot_temporal_convergence_rate():
    # dt_values = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    dt_values = [0.01, 0.005, 0.002, 0.001]
    dt_hifi = 0.0001
    errors = []
    N = 16
    tfinal = 1

    dt = dt_hifi
    sample_interval = int(1.0 / dt)
    output_files_dir = f'output_Ma_0.025_N_{N}_dt_{dt_hifi}_tfinal_{tfinal}_sample_{sample_interval}/'
    idx_tfinal_hifi = int(tfinal / (sample_interval * dt_hifi))

    u_hifi_tfinal = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal_hifi}.txt')
    v_hifi_tfinal = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal_hifi}.txt')
    p_hifi_tfinal = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal_hifi}.txt')

    # high-fidelity solution at final time step

    for dt in dt_values:
        sample_interval = int(1.0 / dt)
        output_files_dir = f'output_Ma_0.025_N_{N}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'
        idx_tfinal = int(tfinal / (sample_interval * dt))

        u_coarse_tfinal = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
        v_coarse_tfinal = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
        p_coarse_tfinal = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')

        # compute L2 error at final time step
        error_u = np.sqrt(np.mean((u_coarse_tfinal - u_hifi_tfinal)**2))
        error_v = np.sqrt(np.mean((v_coarse_tfinal - v_hifi_tfinal)**2))
        error_p = np.sqrt(np.mean((p_coarse_tfinal - p_hifi_tfinal)**2))

        total_error = np.sqrt(error_u**2 + error_v**2 + error_p**2)
        errors.append(total_error)
        print(f"dt = {dt}, idx_tfinal = {idx_tfinal}, L2 Error at tfinal = {total_error}")

    # compute the convergence rate
    p, _ = np.polyfit(np.log(dt_values), np.log(errors), 1)
    print(f"Rate of convergence in time at N = {N}, p = {p}")
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dt_values, errors, 'o-', label='Error')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('L2 Error')
    plt.title(f'Temporal convergence at N = {N}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()

def plot_spatial_convergence_rate():
    # dt_values = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    nx_values = [8, 16, 32]
    nx_hifi = 64
    dx_values = 1.0 / np.array(nx_values)
    
    dt = 1e-03
    sample_interval = 1000
    dt_sample = dt * sample_interval
    tfinal = 1

    coordinate_numer = 24
    coordinate_denom = 32

    output_files_dir = f'output_Ma_0.025_N_{nx_hifi}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'
    idx_tfinal = int(tfinal / (sample_interval * dt))
    
    u_hifi_tfinal = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
    v_hifi_tfinal = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
    p_hifi_tfinal = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')
    u_hifi_tfinal_sample_point = u_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]
    v_hifi_tfinal_sample_point = v_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]
    p_hifi_tfinal_sample_point = p_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]

    # high-fidelity solution at final time step
    errors = []
    
    for nx in nx_values:
        output_files_dir = f'output_Ma_0.025_N_{nx}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'

        u_tfinal = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
        v_tfinal = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
        p_tfinal = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')
        u_tfinal_sample_point = u_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]
        v_tfinal_sample_point = v_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]
        p_tfinal_sample_point = p_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]

        error_u = u_tfinal_sample_point - u_hifi_tfinal_sample_point
        error_v = v_tfinal_sample_point - v_hifi_tfinal_sample_point
        error_p = p_tfinal_sample_point - p_hifi_tfinal_sample_point
        total_error = np.sqrt(error_u**2 + error_v**2 + error_p**2)
        errors.append(total_error)
        print(f"Nx = {nx}, dx = {dx_values[nx_values.index(nx)]}, L2 Error at t = {tfinal}, (x, y) = ({coordinate_numer/coordinate_denom}, {coordinate_numer/coordinate_denom}) = {total_error}")
        
    # compute the convergence rate
    p, _ = np.polyfit(np.log(dx_values), np.log(errors), 1)
    print(f"Rate of convergence in space at dt = {dt}, p = {p}")
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dx_values, errors, 'o-', label='Error')
    plt.xlabel('Spatial grid size (dx)')
    plt.ylabel('L2 Error')
    plt.title(f'Spatial convergence at dt = {dt}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()

def check_whether_accurate_in_time():
    
    # dt_values = [0.01, 0.005, 0.002, 0.001, 0.0001]
    dt_values = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    # dt_values = [1, 2, 5, 10]
    errors = []
    N = 128
    # tfinal = 10
    tfinal = 1
    
    dt = 0.001
    # dt = 1
    sample_interval = int((tfinal + 0.001) / dt)
    output_files_dir = f'output_Ma_0.025_N_{N}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'
    idx_tfinal = int((tfinal + 0.001) / (sample_interval * dt))

    u_tfinal_old = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
    v_tfinal_old = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
    p_tfinal_old = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')

    # dt_values.remove(dt)

    for dt in dt_values:
        sample_interval = int((tfinal + 0.001) / dt)
        output_files_dir = f'output_Ma_0.025_N_{N}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'
        idx_tfinal = int((tfinal + 0.001) / (sample_interval * dt))

        u_tfinal = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
        v_tfinal = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
        p_tfinal = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')

        # compute L2 error at final time step
        error_u = np.sqrt(np.mean((u_tfinal - u_tfinal_old)**2))
        error_v = np.sqrt(np.mean((v_tfinal - v_tfinal_old)**2))
        error_p = np.sqrt(np.mean((p_tfinal - p_tfinal_old)**2))

        total_error = np.sqrt(error_u**2 + error_v**2 + error_p**2)
        errors.append(total_error)
        print(f"dt = {dt}, idx_tfinal = {idx_tfinal}, L2 Error at tfinal = {total_error}")

        u_tfinal_old = u_tfinal
        v_tfinal_old = v_tfinal
        p_tfinal_old = p_tfinal
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dt_values, errors, 'o-', label='Error')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('L2 Error')
    # plt.title(f'Temporal convergence at N = {N}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()
    
def check_whether_accurate_in_space():
    
    nx_values = [8, 16, 32, 64]
    dx_values = 1.0 / np.array(nx_values)
    errors = []
    
    dt = 1e-03
    tfinal = 1
    
    nx = 4
    sample_interval = int(1.0 / dt)
    output_files_dir = f'output_Ma_0.025_N_{nx}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'
    idx_tfinal = int(tfinal / (sample_interval * dt))

    u_tfinal_old = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
    v_tfinal_old = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
    p_tfinal_old = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')
    
    for nx in nx_values:
        output_files_dir = f'output_Ma_0.025_N_{nx}_dt_{dt}_tfinal_{tfinal}_sample_{sample_interval}/'
        idx_tfinal = int(tfinal / (sample_interval * dt))

        u_tfinal = np.loadtxt(f'{output_files_dir}u/u_{idx_tfinal}.txt')
        v_tfinal = np.loadtxt(f'{output_files_dir}v/v_{idx_tfinal}.txt')
        p_tfinal = np.loadtxt(f'{output_files_dir}p/p_{idx_tfinal}.txt')
        
        # Perform averaging coarse graining

        u_tfinal_coarse = u_tfinal.reshape((nx//2, 2, nx//2, 2)).mean(axis=(1, 3))
        v_tfinal_coarse = v_tfinal.reshape((nx//2, 2, nx//2, 2)).mean(axis=(1, 3))
        p_tfinal_coarse = p_tfinal.reshape((nx//2, 2, nx//2, 2)).mean(axis=(1, 3))

        # compute L2 error at final time step
        error_u = np.sqrt(np.mean((u_tfinal_coarse - u_tfinal_old)**2))
        error_v = np.sqrt(np.mean((v_tfinal_coarse - v_tfinal_old)**2))
        error_p = np.sqrt(np.mean((p_tfinal_coarse - p_tfinal_old)**2))

        total_error = np.sqrt(error_u**2 + error_v**2 + error_p**2)
        print(f"Nx = {nx}, dx = {1.0/nx}, L2 Error at tfinal = {total_error}")
        errors.append(total_error)

        u_tfinal_old = u_tfinal
        v_tfinal_old = v_tfinal
        p_tfinal_old = p_tfinal
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dx_values, errors, 's-', label='Pairwise Change (Coarsening)')
    plt.xlabel('Spatial grid size (dx) of coarse grid')
    plt.ylabel('L2 Norm of Difference (||u_coarse - avg(u_fine)||)')
    plt.title(f'Spatial Solution Change (Coarsening, dt = {dt})')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    
    # plot_solution_with_time()

    # plot_temporal_convergence_rate()
    
    # plot_temporal_convergence_rate_Richardson()
    
    # plot_spatial_convergence_rate()
        
    # plot_primary_and_secondary_conservation()
    
    check_whether_accurate_in_time() # results: t = 0.002 is enough when N = 16
    
    # check_whether_accurate_in_space()
    
if __name__ == "__main__":
    main()
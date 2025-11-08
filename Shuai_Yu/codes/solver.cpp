#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <filesystem>

class Parameters {
public:
    // Physical constants
    const double L = 1.0;
    const double Re = 100.0;
    const double Newton_err_tol = 1e-12;
    const int Newton_max_iter = 10;
    const double gamma = 1.4; // specific heat ratio for ideal diatomic gas

    double Ma, dt, dt_sample, dx, dy, p_init, t_final;
    int N, Nt, sample_interval;

    Parameters(double Ma, int N, double dt, double t_final, int sample_interval)
        :Ma(Ma), N(N), dt(dt), t_final(t_final), sample_interval(sample_interval)
    {
        dx = L / (N - 1);
        dy = L / (N - 1);
        dt_sample = dt * sample_interval;
        Nt = 1 + static_cast<int>(std::ceil(t_final / dt));
        p_init = 1.0 / (gamma * Ma * Ma); // initial pressure based on ideal gas law and dimensionless scaling
    }
};

using StateVector = std::vector<Eigen::VectorXd>; // U = dimensionless vectorized physical fields [u1, u2, p]

void compute_tentative_velocity(const StateVector& U, StateVector& U_tmp, const Parameters& params, const double t_current);
    void build_residual_and_jacobian(Eigen::VectorXd& Residual,
                                    Eigen::SparseMatrix<double>& J,
                                    std::vector<Eigen::Triplet<double>>& triplet_list,
                                    const Eigen::VectorXd& u_old,
                                    const Eigen::VectorXd& v_old,
                                    const Eigen::VectorXd& u,
                                    const Eigen::VectorXd& v,
                                    const Parameters& params,
                                    const double t_new);
void solve_PPE(StateVector& U, const StateVector& U_tmp, const Eigen::SparseLU<Eigen::SparseMatrix<double>>& solver, const Parameters& params);
    Eigen::SparseMatrix<double> build_augmented_Laplacian(const Parameters& params);
void update_velocity(StateVector& U, const StateVector& U_tmp, const Parameters& params);
void check_incompressibility(const StateVector& U, const Parameters& params);
void save_vector(const Eigen::VectorXd& vector, const std::string& filename);

inline int k_idx(int i, int j, int N) { // i for x index, j for y index (i for column, j for row)
    return j * N + i;
}

int main(int argc, char* argv[]) {
    // Check the number of inputs

    if (argc != 6) {
        std::cerr << "Error: get " << argc - 1 << " input arguments, expect 5." << std::endl;
        std::cerr << "Usage: ./solver Ma N dt t_final sample_interval" << std::endl;
        return 1; // Error exit
    }

    try {
        double Ma                = std::stod(argv[1]);
        int    N                 = std::stoi(argv[2]);
        double dt                = std::stod(argv[3]);
        double t_final           = std::stod(argv[4]);
        int    sample_interval   = std::stoi(argv[5]);

        std::ostringstream output_folder_name;
        output_folder_name << "output_Ma_" << Ma << "_N_" << N << "_dt_" << dt << "_tfinal_" << t_final << "_sample_" << sample_interval;
        std::filesystem::path output_folder_path = output_folder_name.str();

        // Create the Parameters object
        Parameters params(Ma, N, dt, t_final, sample_interval);

        // Set up the 3-element state vector and vectors for quantities and intermediate state vectors
        StateVector U(2), U_tmp(2); // U = dimensionless vectorized physical fields [u, v, p]

        // Step 1: Initialize the NSE state vector
        // Initial conditions (Eq. 26-28)
        U[0] = Eigen::VectorXd::Zero(2 * params.N * params.N); // u and v are both stored in a single vector for easier Jacobian assembly
        U[1] = Eigen::VectorXd::Zero(params.N * params.N); // actually the initial condition for pressure is not important, here we just make a placeholder
        

        Eigen::SparseMatrix<double> Laplacian_aug = build_augmented_Laplacian(params); // Coefficient matrix for the PPE
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(Laplacian_aug);
        if(solver.info() != Eigen::Success) {
            throw std::runtime_error("Decomposition failed!");
        }

        int counter = 0;

        // Create output directories if they do not exist
        std::filesystem::create_directories(output_folder_path);
        std::filesystem::create_directories(output_folder_path / "u");
        std::filesystem::create_directories(output_folder_path / "v");
        std::filesystem::create_directories(output_folder_path / "p");

        for (int k = 0; k < params.Nt; ++k) {
            double t_current = k * params.dt;
                if (k == 0) {
                std::cout << "--- Simulation Starting ---" << std::endl;
                std::cout << "  Re = " << params.Re << ", Ma = " << params.Ma << std::endl;
                std::cout << "  Grid = " << params.N << "x" << params.N << std::endl;
                std::cout << "  dt = " << params.dt << ", t_final = " << params.t_final << " (Nt = " << params.Nt << ")" << std::endl;
                std::cout << "---------------------------" << std::endl;
            }
            // Output snapshots at specified intervals
            if (k % params.sample_interval == 0) {
                // Create filenames with zero-padded numbers
                std::cout << "Saving snapshot at t* = " << t_current << " (timestep k = " << k << ")" << std::endl;
                std::string filename_u  = (output_folder_path / "u" / ("u_" + std::to_string(counter) + ".txt")).string();
                std::string filename_v  = (output_folder_path / "v" / ("v_" + std::to_string(counter) + ".txt")).string();
                std::string filename_p   = (output_folder_path / "p" / ("p_" + std::to_string(counter) + ".txt")).string();
                // 3. Call your save function for each variable
                //    (Make sure you have created an "output" directory first!)
                save_vector(U[0].head(params.N * params.N), filename_u);
                save_vector(U[0].tail(params.N * params.N), filename_v);
                save_vector(U[1].array() + params.p_init, filename_p); // add the initial pressure back for output
                // 4. Increment the counter for the next snapshot
                counter++;
            }

            // Fractional step method to update the state vector U
            // 1. Solve the tentative velocity field U_tmp without pressure gradient using backward Euler, use the wall BC at t+dt
            U_tmp = U; // temporary state vector for intermediate steps
            compute_tentative_velocity(U, U_tmp, params, t_current);
            solve_PPE(U, U_tmp, solver, params); // 2. Solve the PPE to get pressure at t+dt
            update_velocity(U, U_tmp, params); // 3. Update the velocity field using the pressure gradient
            check_incompressibility(U, params); // check divergence-free condition
            // Check for simulation divergence
            if (!U[0].allFinite() || !U[1].allFinite()) {
                std::cerr << "Error: Simulation diverged at timestep k = " << k << " (t* = " << t_current << ")" << std::endl;
                return 1;
            }            
        }
        std::cout << "Simulation finished successfully." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error when extracting the input arguments: " << e.what() << std::endl;
        std::cerr << "Please check the .sh file, not all arguments are valid numbers with compatible types." << std::endl;
        return 1;
    }
}

void compute_tentative_velocity(const StateVector& U, StateVector& U_tmp, const Parameters& params, double t_current) {
    // Placeholder implementation
    int N = params.N;

    Eigen::SparseMatrix<double> J(2 * N * N, 2 * N * N); // Jacobian matrix for the tentative 2d velocity
    Eigen::VectorXd Residual(2 * N * N); // Right-hand side vector
    Eigen::VectorXd U_diff(2 * N * N); // Update step for [u; v]

    std::vector<Eigen::Triplet<double>> triplet_list;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> iterSolver; // A biconjugate gradient solver with incomplete LU preconditioner to iteratively solve the linear system emerged in the Newton's method

    int iter = 0;

    build_residual_and_jacobian(Residual,
                                    J,
                                    triplet_list,
                                    U[0].head(N * N), // u_old
                                    U[0].tail(N * N), // v_old
                                    U_tmp[0].head(N * N), // u_tmp
                                    U_tmp[0].tail(N * N), // v_tmp
                                    params,
                                    t_current + params.dt // t_new
                                );

    iterSolver.preconditioner().setDroptol(1e-1); // set the drop tolerance for the ILU preconditioner (if the entry in J is smaller than 1e-1, then the preconditioner drops the entry)
    iterSolver.setTolerance(1e-3); // set the tolerance for the iterative solver error
    iterSolver.compute(J);
    if (iterSolver.info() != Eigen::Success) {
        throw std::runtime_error("Preconditioner computation failed.");
    }
    U_diff = iterSolver.solve(-Residual);
    if (iterSolver.info() != Eigen::Success) {
        throw std::runtime_error("Initial linear solve failed.");
    }

    // Update the guesses
    U_tmp[0].head(N * N) += U_diff.head(N * N);
    U_tmp[0].tail(N * N) += U_diff.tail(N * N);

    // Check for convergence
    if (U_diff.norm() < params.Newton_err_tol) {
        std::cout << "time = " << t_current + params.dt << ", Tentative velocity iteration " << iter + 1 << ", Newton method error = " << U_diff.norm() << ", threshold = " << params.Newton_err_tol << ", converged." << std::endl;
        return;
    }
    else {
        std::cout << "time = " << t_current + params.dt << ", Tentative velocity iteration " << iter + 1 << ", Newton method error = " << U_diff.norm() << ", threshold = " << params.Newton_err_tol << ", not converged yet." << std::endl;
        for (int iter = 1; iter < params.Newton_max_iter; ++iter) {
            // Assemble J and Residual based on current u1_tmp and u2_tmp
            build_residual_and_jacobian(Residual,
                                        J,
                                        triplet_list,
                                        U[0].head(N * N), // u_old
                                        U[0].tail(N * N), // v_old
                                        U_tmp[0].head(N * N), // u_tmp
                                        U_tmp[0].tail(N * N), // v_tmp
                                        params,
                                        t_current + params.dt // t_new
                                    );
            // Solve for the update step; the sparsity pattern of J does not change, so we can reuse the symbolic factorization
            iterSolver.compute(J);
            if (iterSolver.info() != Eigen::Success) {
                throw std::runtime_error("Preconditioner computation failed.");
            }
            U_diff = iterSolver.solve(-Residual);
            if (iterSolver.info() != Eigen::Success) {
                throw std::runtime_error("Solving failed in Newton iteration.");
            }

            // Update the guesses
            U_tmp[0].head(N * N) += U_diff.head(N * N);
            U_tmp[0].tail(N * N) += U_diff.tail(N * N);

            // Check for convergence
            if (U_diff.norm() < params.Newton_err_tol) {
                std::cout << "time = " << t_current + params.dt << ", Tentative velocity iteration " << iter + 1 << ", Newton method error = " << U_diff.norm() << ", threshold = " << params.Newton_err_tol << ", converged." << std::endl;
                break;
            }
            else {
                std::cout << "time = " << t_current + params.dt << ", Tentative velocity iteration " << iter + 1 << ", Newton method error = " << U_diff.norm() << ", threshold = " << params.Newton_err_tol << ", not converged yet." << std::endl;
            }
        }
        if (U_diff.norm() >= params.Newton_err_tol) {
            std::cerr << "time = " << t_current + params.dt << ", Warning!!! : Tentative velocity Newton method did not converge within the maximum number of iterations." << std::endl;
        }
    }

}

void build_residual_and_jacobian(Eigen::VectorXd& R,
    Eigen::SparseMatrix<double>& J,
    std::vector<Eigen::Triplet<double>>& triplet,
    const Eigen::VectorXd& u_old,
    const Eigen::VectorXd& v_old,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v,
    const Parameters& params,
    const double t_new
    ) {

    // Build the residual and Jacobian for the tentative velocity step

    int N = params.N;
    double dt = params.dt;
    double dx = params.dx;
    double dy = params.dy;
    double Re = params.Re;

    R.setZero();
    triplet.clear();
    double U_wall = std::sin(2.0 * t_new / Re); // moving lid BC at top wall

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int idx_P = k_idx(i, j, N); // linear index for (i,j)

            // First, we build the neighbor indices with boundary checks
            int idx_E = (i == N - 1) ? -1 : k_idx(i + 1, j, N); // if east point is out of bounds, set to -1, other wise is the point to the right
            int idx_W = (i == 0)     ? -1 : k_idx(i - 1, j, N);
            int idx_N = (j == N - 1) ? -1 : k_idx(i, j + 1, N);
            int idx_S = (j == 0)     ? -1 : k_idx(i, j - 1, N);

            double uP = u[idx_P];
            double vP = v[idx_P];

            double uE = (idx_E == -1) ? -1.0 * uP : u[idx_E];
            double uW = (idx_W == -1) ? -1.0 * uP : u[idx_W];
            double uN = (idx_N == -1) ? (2.0 * U_wall - 1.0 * uP) : u[idx_N];
            double uS = (idx_S == -1) ? -1.0 * uP : u[idx_S];
            double vE = (idx_E == -1) ? -1.0 * vP : v[idx_E];
            double vW = (idx_W == -1) ? -1.0 * vP : v[idx_W];
            double vN = (idx_N == -1) ? -1.0 * vP : v[idx_N];
            double vS = (idx_S == -1) ? -1.0 * vP : v[idx_S];

            double ue = 0.5 * (uP + uE);
            double uw = 0.5 * (uP + uW);
            double un = 0.5 * (uP + uN);
            double us = 0.5 * (uP + uS);
            double ve = 0.5 * (vP + vE);
            double vw = 0.5 * (vP + vW);
            double vn = 0.5 * (vP + vN);
            double vs = 0.5 * (vP + vS);

            double dudxe = (uE - uP) / dx;
            double dudxw = (uP - uW) / dx;
            double dudyn = (uN - uP) / dy;
            double dudys = (uP - uS) / dy;
            double dvdxe = (vE - vP) / dx;
            double dvdxw = (vP - vW) / dx;
            double dvdyn = (vN - vP) / dy;
            double dvdys = (vP - vS) / dy;

            double R_time_derivative_u = (uP - u_old[idx_P]) / dt;
            double R_convection_u = (ue * ue - uw * uw) / dx + (vn * un - vs * us) / dy;
            double R_diffusion_u = (-1.0/Re) * ((dudxe - dudxw) / dx + (dudyn - dudys) / dy);
            double R_time_derivative_v = (vP - v_old[idx_P]) / dt;
            double R_convection_v = (ue * ve - uw * vw) / dx + (vn * vn - vs * vs) / dy;
            double R_diffusion_v = (-1.0/Re) * ((dvdxe - dvdxw) / dx + (dvdyn - dvdys) / dy);

            R[idx_P] = R_time_derivative_u + R_convection_u + R_diffusion_u;
            R[N * N + idx_P] = R_time_derivative_v + R_convection_v + R_diffusion_v;

            double dueduP = (idx_E == -1) ? 0.0 : 0.5; // if at east boundary, then ue = (uP + -uP) / 2 = 0, so derivative is 0, else ue = (uP + uE) / 2, derivative is 0.5
            double duwduP = (idx_W == -1) ? 0.0 : 0.5; // if at west boundary, then uw = (-uP + uP) / 2 = 0, so derivative is 0, else uw = (uP + uW) / 2, derivative is 0.5
            double dunduP = (idx_N == -1) ? 0.0 : 0.5; // if at north boundary, then un = (uP + (2*U_wall - uP)) / 2 = U_wall, so derivative is 0, else un = (uP + uN) / 2, derivative is 0.5
            double dusduP = (idx_S == -1) ? 0.0 : 0.5; // if at south boundary, then us = (-uP + uP) / 2 = 0, so derivative is 0, else us = (uP + uS) / 2, derivative is 0.5
            double dvedvP = (idx_E == -1) ? 0.0 : 0.5; // if at east boundary, then ve = (vP + -vP) / 2 = 0, so derivative is 0, else ve = (vP + vE) / 2, derivative is 0.5
            double dvwdvP = (idx_W == -1) ? 0.0 : 0.5; // if at west boundary, then vw = (vP + -vP) / 2 = 0, so derivative is 0, else vw = (vP + vW) / 2, derivative is 0.5
            double dvndvP = (idx_N == -1) ? 0.0 : 0.5; // if at north boundary, then vn = (vP + -vP) / 2 = 0, so derivative is 0, else vn = (vP + vN) / 2, derivative is 0.5
            double dvsdvP = (idx_S == -1) ? 0.0 : 0.5; // if at south boundary, then vs = (vP + -vP) / 2 = 0, so derivative is 0, else vs = (vP + vS) / 2, derivative is 0.5
            double dueduE = (idx_E == -1) ? 0.0 : 0.5; // if at east boundary, then ue = (uP + -uP) / 2 = 0, so derivative is 0, else ue = (uP + uE) / 2, derivative is 0.5

            double d_dudxe_duP = (idx_E == -1) ? (-2.0/dx) : (-1.0 / dx); // if at east boundary, then dudxe = (-uP - uP) / dx = -2uP/dx, derivative is -2/dx, else dudxe = (uE - uP) / dx, derivative is -1/dx
            double d_dudxw_duP = (idx_W == -1) ? (2.0/dx) : (1.0 / dx); // if at west boundary, then dudxw = (uP - -uP) / dx = 2uP/dx, derivative is 2/dx, else dudxw = (uP - uW) / dx, derivative is 1/dx
            double d_dudyn_duP = (idx_N == -1) ? (-2.0/dy) : (-1.0 / dy); // if at north boundary, then dudyn = (uN - uP) / dy = (2*U_wall - uP - uP)/dy, derivative is -2.0/dy, else dudyn = (uN - uP) / dy, derivative is -1/dy
            double d_dudys_duP = (idx_S == -1) ? (2.0/dy) : (1.0 / dy); // if at south boundary, then dudys = (uP - -uP) / dy = 2uP/dy, derivative is 2/dy, else dudys = (uP - uS) / dy, derivative is 1/dy
            double d_dvdxe_dvP = (idx_E == -1) ? (-2.0/dx) : (-1.0 / dx); // if at east boundary, then dvdxe = (-vP - vP) / dx = -2vP/dx, derivative is -2/dx, else dvdxe = (vE - vP) / dx, derivative is -1/dx
            double d_dvdxw_dvP = (idx_W == -1) ? (2.0/dx) : (1.0 / dx); // if at west boundary, then dvdxw = (vP - -vP) / dx = 2vP/dx, derivative is 2/dx, else dvdxw = (vP - vW) / dx, derivative is 1/dx
            double d_dvdyn_dvP = (idx_N == -1) ? (-2.0/dy) : (-1.0 / dy); // if at north boundary, then dvdyn = (-vP - vP) / dy = -2vP/dy, derivative is -2.0/dy, else dvdyn = (vN - vP) / dy, derivative is -1/dy
            double d_dvdys_dvP = (idx_S == -1) ? (2.0/dy) : (1.0 / dy); // if at south boundary, then dvdys = (vP - -vP) / dy = 2vP/dy, derivative is 2/dy, else dvdys = (vP - vS) / dy, derivative is 1/dy

            // Now, we are going to fill the Jacobian entries
            // First, we fill in the Jacobian entries for the point itself
            double Jacobian_coeff_dR_time_derivative_u_duP = 1.0 / dt;
            double Jacobian_coeff_dR_convection_u_duP      = (2.0 * ue * dueduP - 2.0 * uw * duwduP)/dx + (vn * dunduP - vs * dusduP)/dy;
            double Jacobian_coeff_dR_diffusion_u_duP       = (-1.0/Re) * ((d_dudxe_duP - d_dudxw_duP) / dx + (d_dudyn_duP - d_dudys_duP) / dy);
            double Jacobian_coeff_dR_u_duP = Jacobian_coeff_dR_time_derivative_u_duP
                                            + Jacobian_coeff_dR_convection_u_duP
                                            + Jacobian_coeff_dR_diffusion_u_duP;
            triplet.emplace_back(idx_P, idx_P, Jacobian_coeff_dR_u_duP); // dR_u_p/du_p

            double Jacobian_coeff_dR_time_derivative_u_dvP = 0.0;
            double Jacobian_coeff_dR_convection_u_dvP      = (dvndvP * un - dvsdvP * us)/dy; 
            double Jacobian_coeff_dR_diffusion_u_dvP       = 0.0;
            double Jacobian_coeff_dR_u_dvP = Jacobian_coeff_dR_time_derivative_u_dvP
                                            + Jacobian_coeff_dR_convection_u_dvP
                                            + Jacobian_coeff_dR_diffusion_u_dvP;
            triplet.emplace_back(idx_P, idx_P + N * N, Jacobian_coeff_dR_u_dvP); // dR_u_p/dv_p

            double Jacobian_coeff_dR_time_derivative_v_duP = 0.0;
            double Jacobian_coeff_dR_convection_v_duP      = (dueduP * ve - duwduP * vw)/dx;
            double Jacobian_coeff_dR_diffusion_v_duP       = 0.0;
            double Jacobian_coeff_dR_v_duP = Jacobian_coeff_dR_time_derivative_v_duP
                                            + Jacobian_coeff_dR_convection_v_duP
                                            + Jacobian_coeff_dR_diffusion_v_duP;
            triplet.emplace_back(idx_P + N * N, idx_P, Jacobian_coeff_dR_v_duP); // dR_v_p/du_p

            double Jacobian_coeff_dR_time_derivative_v_dvP = 1.0 / dt;
            double Jacobian_coeff_dR_convection_v_dvP      = (ue * dvedvP - uw * dvwdvP)/dx + (2.0 * vn * dvndvP - 2.0 * vs * dvsdvP)/dy;
            double Jacobian_coeff_dR_diffusion_v_dvP       = (-1.0/Re) * ((d_dvdxe_dvP - d_dvdxw_dvP) / dx + (d_dvdyn_dvP - d_dvdys_dvP) / dy);
            double Jacobian_coeff_dR_v_dvP = Jacobian_coeff_dR_time_derivative_v_dvP
                                            + Jacobian_coeff_dR_convection_v_dvP
                                            + Jacobian_coeff_dR_diffusion_v_dvP;
            triplet.emplace_back(idx_P + N * N, idx_P + N * N, Jacobian_coeff_dR_v_dvP); // dR_v_p/dv_p

            // Next, we fill in the Jacobian entries for the east neighbor if it exists

            if (idx_E != -1) {

                double Jacobian_coeff_dR_time_derivative_u_duE = 0.0;
                double Jacobian_coeff_dR_convection_u_duE      = (2.0 * ue * 0.5) / dx; // it should be (2.0 * ue * dueduE)/dx, but we know that if the east neighbor exists, then dueduE = d(0.5uP + 0.5uE)/duE = 0.5
                double Jacobian_coeff_dR_diffusion_u_duE       = (-1.0/Re) * (1.0 / dx) / dx; // it should be (-1.0/Re) * (d_dudxe_duE) / dx, but we know that if the east neighbor exists, then d_dudxe_duE = d((uE - uP)/dx)/duE = 1.0/dx
                double Jacobian_coeff_dR_u_duE = Jacobian_coeff_dR_time_derivative_u_duE
                                                + Jacobian_coeff_dR_convection_u_duE
                                                + Jacobian_coeff_dR_diffusion_u_duE;
                triplet.emplace_back(idx_P, idx_E, Jacobian_coeff_dR_u_duE); // dR_u_p/du_E

                // d_R_u_p/dv_E = 0

                double Jacobian_coeff_dR_time_derivative_v_duE = 0.0;
                double Jacobian_coeff_dR_convection_v_duE      = 0.5 * ve / dx; // it should be (dueduE * ve)/dx, but we know that if the east neighbor exists, then dueduE = d(0.5uP + 0.5uE)/duE = 0.5
                double Jacobian_coeff_dR_diffusion_v_duE       = 0.0;
                double Jacobian_coeff_dR_v_duE = Jacobian_coeff_dR_time_derivative_v_duE
                                                + Jacobian_coeff_dR_convection_v_duE
                                                + Jacobian_coeff_dR_diffusion_v_duE;
                triplet.emplace_back(idx_P + N * N, idx_E, Jacobian_coeff_dR_v_duE); // dR_v_p/du_E

                double Jacobian_coeff_dR_time_derivative_v_dvE = 0.0;
                double Jacobian_coeff_dR_convection_v_dvE      = 0.5 * ue / dx; // it should be (ue * dvedvE)/dx, but we know that if the east neighbor exists, then dvedvE = d(0.5vP + 0.5vE)/dvE = 0.5
                double Jacobian_coeff_dR_diffusion_v_dvE       = (-1.0/Re) * (1.0 / dx) / dx; // it should be (-1.0/Re) * (d_dvdxe_dvE) / dx, but we know that if the east neighbor exists, then d_dvdxe_dvE = d((vE - vP)/dx)/dvE = 1.0/dx
                double Jacobian_coeff_dR_v_dvE = Jacobian_coeff_dR_time_derivative_v_dvE
                                                + Jacobian_coeff_dR_convection_v_dvE
                                                + Jacobian_coeff_dR_diffusion_v_dvE;
                triplet.emplace_back(idx_P + N * N, idx_E + N * N, Jacobian_coeff_dR_v_dvE); // dR_v_p/dv_p

            }

            // Next, we fill in the Jacobian entries for the west neighbor if it exists

            if (idx_W != -1) {

                double Jacobian_coeff_dR_time_derivative_u_duW = 0.0;
                double Jacobian_coeff_dR_convection_u_duW      = (-2.0 * uw * 0.5) / dx; // it should be (-2.0 * uw * duwduW)/dx, but we know that if the west neighbor exists, then duwduW = d(0.5uP + 0.5uW)/duW = 0.5
                double Jacobian_coeff_dR_diffusion_u_duW       = (-1.0/Re) * (1.0 / dx) / dx; // it should be (-1.0/Re) * (-d_dudxw_duW) / dx, but we know that if the west neighbor exists, then d_dudxw_duW = d((uP - uW)/dx)/duW = -1.0/dx
                double Jacobian_coeff_dR_u_duW = Jacobian_coeff_dR_time_derivative_u_duW
                                                + Jacobian_coeff_dR_convection_u_duW
                                                + Jacobian_coeff_dR_diffusion_u_duW;
                triplet.emplace_back(idx_P, idx_W, Jacobian_coeff_dR_u_duW); // dR_u_p/du_W

                // d_R_u_p/dv_W = 0

                double Jacobian_coeff_dR_time_derivative_v_duW = 0.0;
                double Jacobian_coeff_dR_convection_v_duW      = -0.5 * vw / dx; // it should be (-duwduW * vw)/dx, but we know that if the west neighbor exists, then duwduW = d(0.5uP + 0.5uW)/duW = 0.5
                double Jacobian_coeff_dR_diffusion_v_duW       = 0.0;
                double Jacobian_coeff_dR_v_duW = Jacobian_coeff_dR_time_derivative_v_duW
                                                + Jacobian_coeff_dR_convection_v_duW
                                                + Jacobian_coeff_dR_diffusion_v_duW;
                triplet.emplace_back(idx_P + N * N, idx_W, Jacobian_coeff_dR_v_duW); // dR_v_p/du_W

                double Jacobian_coeff_dR_time_derivative_v_dvW = 0.0;
                double Jacobian_coeff_dR_convection_v_dvW      = -0.5 * uw / dx; // it should be (-uw * dvwdvW)/dx, but we know that if the west neighbor exists, then dvwdvW = d(0.5vP + 0.5vW)/dvW = 0.5
                double Jacobian_coeff_dR_diffusion_v_dvW       = (-1.0/Re) * (1.0 / dx) / dx; // it should be (-1.0/Re) * (-d_dvdxw_dvW) / dx, but we know that if the east neighbor exists, then d_dvdxw_dvW = d((vP - vW)/dx)/dvW = -1.0/dx
                double Jacobian_coeff_dR_v_dvW = Jacobian_coeff_dR_time_derivative_v_dvW
                                                + Jacobian_coeff_dR_convection_v_dvW
                                                + Jacobian_coeff_dR_diffusion_v_dvW;
                triplet.emplace_back(idx_P + N * N, idx_W + N * N, Jacobian_coeff_dR_v_dvW); // dR_v_p/dv_W

            }

            // Next, we fill in the Jacobian entries for the north neighbor if it exists

            if (idx_N != -1) {

                double Jacobian_coeff_dR_time_derivative_u_duN = 0.0;
                double Jacobian_coeff_dR_convection_u_duN      = 0.5 * vn / dy; // it should be (vn * dunduN)/dy, but we know that if the north neighbor exists, then dunduN = d(0.5uP + 0.5uN)/duN = 0.5
                double Jacobian_coeff_dR_diffusion_u_duN       = (-1.0/Re) * (1.0 / dy) / dy; // it should be (-1.0/Re) * (d_dudyn_duN) / dy, but we know that if the north neighbor exists, then d_dudyn_duN = d((uN - uP)/dy)/duN = 1.0/dy
                double Jacobian_coeff_dR_u_duN = Jacobian_coeff_dR_time_derivative_u_duN
                                                + Jacobian_coeff_dR_convection_u_duN
                                                + Jacobian_coeff_dR_diffusion_u_duN;
                triplet.emplace_back(idx_P, idx_N, Jacobian_coeff_dR_u_duN); // dR_u_p/du_N

                double Jacobian_coeff_dR_time_derivative_u_dvN = 0.0;
                double Jacobian_coeff_dR_convection_u_dvN      = 0.5 * un / dy; // it should be (dvndvN * un)/dy, but we know that if the north neighbor exists, then dvndvN = d(0.5vP + 0.5vN)/dvN = 0.5
                double Jacobian_coeff_dR_diffusion_u_dvN       = 0.0;
                double Jacobian_coeff_dR_u_dvN = Jacobian_coeff_dR_time_derivative_u_dvN
                                                + Jacobian_coeff_dR_convection_u_dvN
                                                + Jacobian_coeff_dR_diffusion_u_dvN;
                triplet.emplace_back(idx_P, idx_N + N * N, Jacobian_coeff_dR_u_dvN); // dR_u_p/dv_N

                // dR_v_p/du_N = 0

                double Jacobian_coeff_dR_time_derivative_v_dvN = 0.0;
                double Jacobian_coeff_dR_convection_v_dvN      = vn / dy; // it should be (2.0 * vn * dvndvN)/dy, but we know that if the north neighbor exists, then dvndvN = d(0.5vP + 0.5vN)/dvN = 0.5
                double Jacobian_coeff_dR_diffusion_v_dvN       = (-1.0/Re) * (1.0 / dy) / dy; // it should be (-1.0/Re) * (d_dvdyn_dvN) / dy, but we know that if the north neighbor exists, then d_dvdyn_dvN = d((vN - vP)/dy)/dvN = 1.0/dy
                double Jacobian_coeff_dR_v_dvN = Jacobian_coeff_dR_time_derivative_v_dvN
                                                + Jacobian_coeff_dR_convection_v_dvN
                                                + Jacobian_coeff_dR_diffusion_v_dvN;
                triplet.emplace_back(idx_P + N * N, idx_N + N * N, Jacobian_coeff_dR_v_dvN); // dR_v_p/dv_N
            }

            // Next, we fill in the Jacobian entries for the south neighbor if it exists

            if (idx_S != -1) {

                double Jacobian_coeff_dR_time_derivative_u_duS = 0.0;
                double Jacobian_coeff_dR_convection_u_duS      = -0.5 * vs / dy; // it should be (-vs * dusduS)/dy, but we know that if the south neighbor exists, then dusduS = d(0.5uP + 0.5uS)/duS = 0.5
                double Jacobian_coeff_dR_diffusion_u_duS       = (-1.0/Re) * (1.0 / dy) / dy; // it should be (-1.0/Re) * (-d_dudys_duS) / dy, but we know that if the south neighbor exists, then d_dudys_duS = d((uP - uS)/dy)/duS = -1.0/dy
                double Jacobian_coeff_dR_u_duS = Jacobian_coeff_dR_time_derivative_u_duS
                                                + Jacobian_coeff_dR_convection_u_duS
                                                + Jacobian_coeff_dR_diffusion_u_duS;
                triplet.emplace_back(idx_P, idx_S, Jacobian_coeff_dR_u_duS); // dR_u_p/du_S

                double Jacobian_coeff_dR_time_derivative_u_dvS = 0.0;
                double Jacobian_coeff_dR_convection_u_dvS      = -0.5 * us / dy; // it should be (-us * dvsdvS)/dy, but we know that if the south neighbor exists, then dvsdvS = d(0.5vP + 0.5vS)/duS = 0.5
                double Jacobian_coeff_dR_diffusion_u_dvS       = 0.0;
                double Jacobian_coeff_dR_u_dvS = Jacobian_coeff_dR_time_derivative_u_dvS
                                                + Jacobian_coeff_dR_convection_u_dvS
                                                + Jacobian_coeff_dR_diffusion_u_dvS;
                triplet.emplace_back(idx_P, idx_S + N * N, Jacobian_coeff_dR_u_dvS); // dR_u_p/dv_S

                // dR_v_p/du_S = 0

                double Jacobian_coeff_dR_time_derivative_v_dvS = 0.0;
                double Jacobian_coeff_dR_convection_v_dvS      = -vs / dy; // it should be (-2 * vs * dvsdvS)/dy, but we know that if the south neighbor exists, then dvsdvS = d(0.5*vP + 0.5*vS)/dvS = 0.5
                double Jacobian_coeff_dR_diffusion_v_dvS       = (-1.0/Re) * (1.0 / dy) / dy; // it should be (-1.0/Re) * (-d_dvdys_dvS) / dy, but we know that if the south neighbor exists, then d_dvdys_dvS = d((vP - vS)/dy)/dvS = -1.0/dy
                double Jacobian_coeff_dR_v_dvS = Jacobian_coeff_dR_time_derivative_v_dvS
                                                + Jacobian_coeff_dR_convection_v_dvS
                                                + Jacobian_coeff_dR_diffusion_v_dvS;
                triplet.emplace_back(idx_P + N * N, idx_S + N * N, Jacobian_coeff_dR_v_dvS); // dR_v_p/dv_S
            }

        }
    }

    J.setFromTriplets(triplet.begin(), triplet.end());

}


void solve_PPE(StateVector& U, const StateVector& U_tmp, const Eigen::SparseLU<Eigen::SparseMatrix<double>>& solver, const Parameters& params) {
    // In this equation, we try to solve for the "zero-mean" pressure p, such that the velocity field is divergence-free
    // the reason why we need zero mean is because the pressure is only defined up to a constant in incompressible flows

    // To solve the pressure, we need to solve the augmented Poisson equation:
    // | L   1_{N^2}   | |p|   = | rhs |
    // | 1_{N^2}^T   0 | |λ|     |  0  |
    // where L is the discrete Laplacian operator with Neumann boundary conditions for pressure,
    // rhs = (1/dt) * div(u*), and λ is the Lagrange multiplier for the zero-mean constraint.
    // Ideally, lambda = sum(rhs) / N^2 = 0, since rhs should have zero mean.
    int N = params.N; 
    Eigen::VectorXd rhs(N * N + 1); // the divergence of the tentative velocity field
    rhs[N * N] = 0.0; // the last entry corresponds to the constraint for zero-mean pressure

    Eigen::VectorXd u = U_tmp[0].head(N * N);
    Eigen::VectorXd v = U_tmp[0].tail(N * N);

    // Build the right-hand side vector
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int idx_P = k_idx(i, j, N); // linear index for (i,j) i for x and j for y

            // First, we build the neighbor indices with boundary checks
            int idx_E = (i == N - 1) ? -1 : k_idx(i + 1, j, N);
            int idx_N = (j == N - 1) ? -1 : k_idx(i, j + 1, N);
            int idx_W = (i == 0)     ? -1 : k_idx(i - 1, j, N);
            int idx_S = (j == 0)     ? -1 : k_idx(i, j - 1, N);
            
            double ue = (idx_E == -1) ? 0.0 : 0.5 * (u[idx_E] + u[idx_P]);
            double vn = (idx_N == -1) ? 0.0 : 0.5 * (v[idx_N] + v[idx_P]);
            double uw = (idx_W == -1) ? 0.0 : 0.5 * (u[idx_W] + u[idx_P]);
            double vs = (idx_S == -1) ? 0.0 : 0.5 * (v[idx_S] + v[idx_P]);

            rhs[idx_P] = (1.0/params.dt) * ((ue - uw) / params.dx + (vn - vs) / params.dy); // divergence at point P, excluding the fixed pressure point

        }
    }

    // Check that the rhs should have zero mean
    double mean_rhs = rhs.mean();
    std::cout << "Mean of RHS in PPE: " << mean_rhs << std::endl;
    if (std::abs(mean_rhs) > 1e-8) {
        throw std::runtime_error("RHS of PPE does not have zero mean!");
    }
    // Solve the linear system
    U[1] = solver.solve(rhs).head(N * N); // extract only the pressure part, ignore the lambda part
    std::cout << "Mean of pressure: " << U[1].mean() << std::endl;
    if(solver.info() != Eigen::Success) {
        throw std::runtime_error("Solving failed!");
    }
    // Check whether the regularizer lambda = 0
    double lambda = solver.solve(rhs)[N * N];
    std::cout << "Lagrange multiplier (should be close to 0): " << lambda << std::endl;
    if (std::abs(lambda) > 1e-8) {
        throw std::runtime_error("Lagrange multiplier for zero-mean pressure is not close to zero!");
    }
}

Eigen::SparseMatrix<double> build_augmented_Laplacian(const Parameters& params) {
    // Build the augmented Laplacian matrix with Neumann boundary conditions for pressure.
    // The final matrix looks like:
    // | L   1_{N^2}   |
    // | 1_{N^2}^T   0 |
    // this is because we need to enforce the zero-mean constraint on pressure to ensure uniqueness of the pressure solution.
    // So the augmented PPE becomes:
    // | L   1_{N^2}   | |p|   = | rhs |
    // | 1_{N^2}^T   0 | |λ|     |  0  |
    // where rhs = (1/dt) * div(u*), and λ is the Lagrange multiplier for the zero-mean constraint.
    // Ideally, lambda = sum(rhs) / N^2 = 0, since rhs should have zero mean.
    int N = params.N;
    double dx = params.dx;
    double dy = params.dy;

    Eigen::SparseMatrix<double> L_original(N * N, N * N);
    Eigen::SparseMatrix<double> L(N * N + 1, N * N + 1);
    std::vector<Eigen::Triplet<double>> triplet_list;

    triplet_list.clear();

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int idx_P = k_idx(i, j, N); // linear index for (i,j)

            // First, we build the neighbor indices with boundary checks
            int idx_E = (i == N - 1) ? -1 : k_idx(i + 1, j, N); // if east point is out of bounds, set to -1, other wise is the point to the right
            int idx_N = (j == N - 1) ? -1 : k_idx(i, j + 1, N);
            int idx_W = (i == 0)     ? -1 : k_idx(i - 1, j, N);
            int idx_S = (j == 0)     ? -1 : k_idx(i, j - 1, N);

            int idx_EE = (i == N - 2) ? -2 : ((i == N - 1) ? -1 : k_idx(i + 2, j, N)); // if second east point is out of bounds, set to -2, if east point is out of bounds, set to -1, otherwise is the point two to the right
            int idx_NN = (j == N - 2) ? -2 : ((j == N - 1) ? -1 : k_idx(i, j + 2, N));
            int idx_WW = (i == 1)     ? -2 : ((i == 0)     ? -1 : k_idx(i - 2, j, N));
            int idx_SS = (j == 1)     ? -2 : ((j == 0)     ? -1 : k_idx(i, j - 2, N));

            // Recall that, our Laplacian scheme is as follows:
            // (p_EE - 2p_P + p_WW) / (4dx^2) + (p_NN - 2p_P + p_SS) / (4dy^2) = rhs_P
            // So we need to consider different cases for boundary conditions

            double d_Laplace_p_dpP  = -2.0 / (4.0 * dx * dx) - 2.0 / (4.0 * dy * dy);
            triplet_list.emplace_back(idx_P, idx_P, d_Laplace_p_dpP);

            if (idx_EE == -1) {// if the first east neighbor is out of bounds, i = Nx - 1, then p_EE = p_W (or p_Nx+1 = p_Nx-2)(Neumann BC at east boundary = p_Nx - p_Nx-2 + p_Nx+1 - p_Nx-1 = 0. Here we choose p_Nx = p_Nx-1 and p_Nx+1 = p_Nx-2)
                // In this case, the Laplace operator becomes:
                // (p_W - 2p_P + p_WW) / (4dx^2) + ...
                double d_Laplace_p_dpW = 1.0 / (4.0 * dx * dx);
                triplet_list.emplace_back(idx_P, idx_W, d_Laplace_p_dpW);
            } else if (idx_EE == -2) {// if the second east neighbor is out of bounds, i = Nx - 2, then we know that p_EE = p_E or p_Nx = p_Nx-1 (Neumann BC at east boundary)
                // In this case, the Laplace operator becomes:
                // (p_E - 2p_P + p_WW) / (4dx^2) + ...
                double d_Laplace_p_dpE = 1.0 / (4.0 * dx * dx);
                triplet_list.emplace_back(idx_P, idx_E, d_Laplace_p_dpE);
                if (idx_WW == -1) {
                    throw std::runtime_error("Too coarse grid: The west neighbor is out of domain when its second east neighbor is out of domain. Grid size N must be equal or larger than 4!");
                }
            } else {
                // In this case, the Laplace operator becomes normal:
                // (p_EE - 2p_P + p_WW) / (4dx^2) + ...
                double d_Laplace_p_dpEE = 1.0 / (4.0 * dx * dx);
                triplet_list.emplace_back(idx_P, idx_EE, d_Laplace_p_dpEE);
            }

            if (idx_WW == -1) {// if the west neighbor is out of bounds, i = 0, then p_WW = p_E (or p_1 = p_-2)(Neumann BC at west boundary = p_0 - p_-2 + p_1 - p_-1 = 0. Here we choose p_0 = p_-1 and p_1 = p_-2)
                // In this case, the Laplace operator becomes:
                // (p_EE - 2p_P + p_E) / (4dx^2) + ...
                double d_Laplace_p_dpE = 1.0 / (4.0 * dx * dx);
                triplet_list.emplace_back(idx_P, idx_E, d_Laplace_p_dpE);
            } else if (idx_WW == -2) {// if the second west neighbor is out of bounds, i = 1, then we know that p_WW = p_-1 = p_0 = p_W
                // In this case, the Laplace operator becomes:
                // (p_EE - 2p_P + p_W) / (4dx^2) + ...
                double d_Laplace_p_dpW = 1.0 / (4.0 * dx * dx);
                triplet_list.emplace_back(idx_P, idx_W, d_Laplace_p_dpW);
                if (idx_EE == -1) {
                    throw std::runtime_error("Too coarse grid: The east neighbor is out of domain when its second west neighbor is out of domain. Grid size N must be equal or larger than 4!");
                }
            } else {
                // In this case, the Laplace operator becomes normal:
                // (p_EE - 2p_P + p_WW) / (4dx^2) + ...
                double d_Laplace_p_dpWW = 1.0 / (4.0 * dx * dx);
                triplet_list.emplace_back(idx_P, idx_WW, d_Laplace_p_dpWW);
            }

            if (idx_NN == -1) {// if the first north neighbor is out of bounds, j = Ny - 1, then p_NN = p_S (or p_Ny-2 = p_Ny+1)(Neumann BC at north boundary = p_Ny - p_Ny-2 + p_Ny+1 - p_Ny-1 = 0. Here we choose p_Ny = p_Ny-1 and p_Ny+1 = p_Ny-2)
                // In this case, the Laplace operator becomes:
                // (p_S - 2p_P + p_SS) / (4dy^2) + ...
                double d_Laplace_p_dpS = 1.0 / (4.0 * dy * dy);
                triplet_list.emplace_back(idx_P, idx_S, d_Laplace_p_dpS);
            } else if (idx_NN == -2) {// if the second north neighbor is out of bounds, j = Ny - 2, then we know that p_NN = p_Ny = p_Ny-1 = p_N (Neumann BC at north boundary)
                // In this case, the Laplace operator becomes:
                // (p_N - 2p_P + p_SS) / (4dy^2) + ...
                double d_Laplace_p_dpN = 1.0 / (4.0 * dy * dy);
                triplet_list.emplace_back(idx_P, idx_N, d_Laplace_p_dpN);
                if (idx_SS == -1) {
                    throw std::runtime_error("Too coarse grid: The south neighbor is out of domain when its second north neighbor is out of domain. Grid size N must be equal or larger than 4!");
                }
            } else {
                // In this case, the Laplace operator becomes normal:
                // (p_NN - 2p_P + p_SS) / (4dy^2) + ...
                double d_Laplace_p_dpNN = 1.0 / (4.0 * dy * dy);
                triplet_list.emplace_back(idx_P, idx_NN, d_Laplace_p_dpNN);
            }

            if (idx_SS == -1) {// if the south neighbor is out of bounds, j = 0, then p_SS = p_N (or p_1 = p_-2)(Neumann BC at south boundary = p_0 - p_-2 + p_1 - p_-1 = 0. Here we choose p_0 = p_-1 and p_1 = p_-2)
                // In this case, the Laplace operator becomes:
                // (p_NN - 2p_P + p_N) / (4dy^2) + ...
                double d_Laplace_p_dpN = 1.0 / (4.0 * dy * dy);
                triplet_list.emplace_back(idx_P, idx_N, d_Laplace_p_dpN);
            } else if (idx_SS == -2) {// if the second south neighbor is out of bounds, i = 1, then we know that p_SS = p_-1 = p_0 = p_S
                // In this case, the Laplace operator becomes:
                // (p_NN - 2p_P + p_S) / (4dy^2) + ...
                double d_Laplace_p_dpS = 1.0 / (4.0 * dy * dy);
                triplet_list.emplace_back(idx_P, idx_S, d_Laplace_p_dpS);
                if (idx_NN == -1) {
                    throw std::runtime_error("Too coarse grid: The north neighbor is out of domain when its second south neighbor is out of domain. Grid size N must be equal or larger than 4!");
                }
            } else {
                // In this case, the Laplace operator becomes normal:
                // (p_NN - 2p_P + p_SS) / (4dy^2) + ...
                double d_Laplace_p_dpSS = 1.0 / (4.0 * dy * dy);
                triplet_list.emplace_back(idx_P, idx_SS, d_Laplace_p_dpSS);
            }
        }
    }

    L_original.setFromTriplets(triplet_list.begin(), triplet_list.end());
    // Test whether our Laplacian satisfy the zero-row-sum property
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(L_original.cols());
    double norm = (L_original * ones).norm();
    // std::cout << "norm of the summation of each row of Laplacian (should be close to 0): " << norm << std::endl;
    if (norm > 1e-8) {
        throw std::runtime_error("Error: Laplacian does not satisfy zero-row-sum property!");
    }
    // Test whether our Laplacian is symmetric
    // Eigen::SparseMatrix<double> L_transpose = L_original.transpose();
    // Eigen::SparseMatrix<double> L_diff = L_original - L_transpose;
    // double sym_norm = L_diff.norm();
    // std::cout << "Symmetry norm of Laplacian: " << sym_norm << std::endl;
    // if (sym_norm > 1e-8) {
    //     throw std::runtime_error("Error: Laplacian is not symmetric!");
    // }

    // if we are good, then continue to build the augmented matrix
    // Last but not least, we need to add the zero-mean constraint for pressure.
    for (int i = 0; i < N * N; ++i) {
        triplet_list.emplace_back(i, N * N, 1.0); // last column
        triplet_list.emplace_back(N * N, i, 1.0); // last row
    }
    L.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return L;
}

void update_velocity(StateVector& U, const StateVector& U_tmp, const Parameters& params) {
    // Update the velocity field using the pressure field: u_new - u_tmp = dt * -grad(p)
    int N = params.N;
    double dt = params.dt;
    double dx = params.dx;
    double dy = params.dy;

    Eigen::VectorXd u_tmp = U_tmp[0].head(N * N);
    Eigen::VectorXd v_tmp = U_tmp[0].tail(N * N);

    Eigen::VectorXd p = U[1];

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int idx_P = k_idx(i, j, N); // linear index for (i,j)

            // First, we build the neighbor indices with boundary checks
            int idx_E = (i == N - 1) ? -1 : k_idx(i + 1, j, N); // if east point is out of bounds, set to -1, other wise is the point to the right
            int idx_W = (i == 0)     ? -1 : k_idx(i - 1, j, N);
            int idx_N = (j == N - 1) ? -1 : k_idx(i, j + 1, N);
            int idx_S = (j == 0)     ? -1 : k_idx(i, j - 1, N);

            double pe = (idx_E == -1) ? p[idx_P] : (p[idx_P] + p[idx_E]) / 2;
            double pw = (idx_W == -1) ? p[idx_P] : (p[idx_P] + p[idx_W]) / 2;
            double pn = (idx_N == -1) ? p[idx_P] : (p[idx_P] + p[idx_N]) / 2;
            double ps = (idx_S == -1) ? p[idx_P] : (p[idx_P] + p[idx_S]) / 2;

            U[0][idx_P] = u_tmp[idx_P] - (dt / dx) * (pe - pw);
            U[0][idx_P + N * N] = v_tmp[idx_P] - (dt / dy) * (pn - ps);
        }
    }
}

void check_incompressibility(const StateVector& U, const Parameters& params) {
    // Check the incompressibility of the velocity field by computing the divergence at each grid point
    int N = params.N;
    Eigen::VectorXd u = U[0].head(N * N);
    Eigen::VectorXd v = U[0].tail(N * N);

    double max_divergence = 0.0;
    double mean_divergence = 0.0;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int idx_P = k_idx(i, j, N); // linear index for (i,j)

            // First, we build the neighbor indices with boundary checks
            int idx_E = (i == N - 1) ? -1 : k_idx(i + 1, j, N);
            int idx_N = (j == N - 1) ? -1 : k_idx(i, j + 1, N);
            int idx_W = (i == 0)     ? -1 : k_idx(i - 1, j, N);
            int idx_S = (j == 0)     ? -1 : k_idx(i, j - 1, N);
            
            double ue = (idx_E == -1) ? 0.0 : 0.5 * (u[idx_E] + u[idx_P]);
            double vn = (idx_N == -1) ? 0.0 : 0.5 * (v[idx_N] + v[idx_P]);
            double uw = (idx_W == -1) ? 0.0 : 0.5 * (u[idx_W] + u[idx_P]);
            double vs = (idx_S == -1) ? 0.0 : 0.5 * (v[idx_S] + v[idx_P]);

            double divergence = ((ue - uw) / params.dx + (vn - vs) / params.dy);
            max_divergence = std::max(max_divergence, std::abs(divergence));
            mean_divergence += divergence / (N * N);
        }
    }

    std::cout << "Maximum divergence in the velocity field: " << max_divergence << std::endl;
    std::cout << "Mean divergence in the velocity field: " << mean_divergence << std::endl;
    if (max_divergence > 1e-12) {
        throw std::runtime_error("Velocity field is not incompressible!");
    }
}


// save_vector: Function to save a vector to a file
void save_vector(const Eigen::VectorXd& vector, const std::string& filename) {
    int nx = static_cast<int>(std::sqrt(vector.size()));
    int ny = nx;
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        matrix_view(vector.data(), ny, nx);
    const static Eigen::IOFormat TXTFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::setprecision(15);
        file << matrix_view.format(TXTFormat);
        file.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}


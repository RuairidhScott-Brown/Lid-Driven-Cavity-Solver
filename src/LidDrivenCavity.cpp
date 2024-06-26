#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>

#define IDX(I,J) ((I)*m_Nx + (J))

#include "../include/LidDrivenCavity.h"
#include "../include/SolverCG.h"


/**
 * @brief Construct a new Lid Driven Cavity::LidDrivenCavity object.
 * 
 * @param    comm                MPI communicator to use.
 */
LidDrivenCavity::LidDrivenCavity(MPI_Comm comm)
{
    m_comm = comm;
}


/**
 * @brief Destroy the Lid Driven Cavity::LidDrivenCavity object and
 * deletes the dynamically allocated arrays.
 * 
 */
LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}


/**
 * @brief Set the domain size of the solver.
 * 
 * @param    xlen                Length in x.
 * @param    ylen                Length in y.
 */
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    m_Lx = xlen;
    m_Ly = ylen;
    UpdateDxDy();
}


/**
 * @brief Set the grid size for the solver domain.
 * 
 * @param    nx                  Number of poinst in x-direction.
 * @param    ny                  Number of points in the y-direction.
 */
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    m_Nx = nx;
    m_Ny = ny;
    UpdateDxDy();
}


/**
 * @brief Set the time step.
 * 
 * @param    deltat              Time step.
 */
void LidDrivenCavity::SetTimeStep(double deltat)
{
    m_dt = deltat;
}


/**
 * @brief Set the final time.
 * 
 * @param    finalt              Final time.
 */
void LidDrivenCavity::SetFinalTime(double finalt)
{
    m_T = finalt;
}


/**
 * @brief Set the Reynolds number.
 * 
 * @param    re                  My Param doc
 */
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    m_Re = re;
    m_nu = 1.0 / re;
}


/**
 * @brief Initialise the lid driven cavity solver by 
 * defining all the variables needed based off of the configuration
 * given to the object.
 * 
 */
void LidDrivenCavity::Initialise()
{
    CleanUp();

    m_v = new double[m_Npts]();
    m_vnew = new double[m_Npts]();
    m_vtemp = new double[m_Npts]();
    m_s = new double[m_Npts]();
    m_tmp = new double[m_Npts]();

    m_cg = new SolverCG(m_Nx, m_Ny, m_dx, m_dy);

    SetSize();
}


/**
 * @brief Performs the integration needed to fine the vorticity and
 * stream function at the end time specificed by T.
 * 
 */
void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(m_T / m_dt);

    for (int t = 0; t < NSteps; ++t) {
        if (m_rank == 0) {
            std::cout << "Step: " << setw(8) << t
                    << "  Time: " << setw(8) << t*m_dt
                    << std::endl;
        }
        Advance();
    }
    // Gather the results for vorticity and the stream function into there respective matrices.
    MPI_Allgatherv(&m_localArray3[m_returnStart], m_returnLength, MPI_DOUBLE, m_vnew, m_returnLengths, m_returnDisplacements, MPI_DOUBLE, m_comm);
    MPI_Allgatherv(&m_localArray1[m_returnStart], m_returnLength, MPI_DOUBLE, m_s, m_returnLengths, m_returnDisplacements, MPI_DOUBLE, m_comm);
}


/**
 * @brief Conver the stream function to the U velocity component.
 * 
 * @param    u                   My Param doc
 */
void LidDrivenCavity::ConvertStreamFunctionToVelocityU(double *const u)
{
    for (int j = 1; j < m_Nx - 1; ++j) {
        for (int i = 1; i < m_Ny - 1; ++i) {
            u[IDX(i, j)] = (m_s[IDX(i+1, j)] - m_s[IDX(i, j)]) / m_dy;
        }
    }
    for (int j = 0; j < m_Nx; ++j) {
        u[IDX(m_Ny-1, j)] = m_U;
    }
}


/**
 * @brief Convert the stream function to V velocity component.
 * 
 * @param    v                   V velocity component.
 */
void LidDrivenCavity::ConvertStreamFunctionToVelocityV(double *const v)
{
    for (int j = 1; j < m_Nx - 1; ++j) {
        for (int i = 1; i < m_Ny - 1; ++i) {
            v[IDX(i, j)] = -(m_s[IDX(i, j+1)] - m_s[IDX(i, j)]) / m_dx;
        }
    }
}


/**
 * @brief Write the vorticity, stream function, U velocity componenet
 * and the V velocity componenet to a file. The files is saved in the
 * local directory.
 * 
 * @param    file                File name.
 */
void LidDrivenCavity::WriteSolution(std::string file)
{
    std::cout << std::fixed << std::setprecision(6);

    if (m_rank != 0)
        return;

    

    double *u0 = new double[m_Nx * m_Ny]();
    double *u1 = new double[m_Nx * m_Ny]();

    ConvertStreamFunctionToVelocityU(u0);
    ConvertStreamFunctionToVelocityV(u1);

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int j = 0; j < m_Nx; ++j) {
        for (int i = 0; i < m_Ny; ++i) {
            k = IDX(i, j);
            f << j * m_dx << " " << i * m_dy << " " << m_vnew[k] << " " << m_s[k]
              << " " << u0[k] << " " << u1[k] << std::endl;
        }
        f << std::endl;
    }
    f.close();

    delete[] u0;
    delete[] u1;
}


/**
 * @brief Prints the current configuration of the lid driven cavity solver. This
 * includes the grid size and constants assocaited with the problem such as the
 * Reynolds number. This should be printed at the start of the computation as
 * it contains a check to make sure the time-step restriction is satisfied.
 * 
 * @return LidDrivenCavityConfigError 
 */
LidDrivenCavityConfigError 
LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << m_Nx << " x " << m_Ny << endl;
    cout << "Spacing:   " << m_dx << " x " << m_dy << endl;
    cout << "Length:    " << m_Lx << " x " << m_Ly << endl;
    cout << "Grid pts:  " << m_Npts << endl;
    cout << "Timestep:  " << m_dt << endl;
    cout << "Steps:     " << ceil(m_T / m_dt) << endl;
    cout << "Reynolds number: " << m_Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << "Size:      " << m_size << endl;
    cout << endl;
    return CheckConfiguration();
}


/**
 * @brief Checks that the time-step restriction is satisfied.
 * 
 * @return LidDrivenCavityConfigError 
 */
LidDrivenCavityConfigError 
LidDrivenCavity::CheckConfiguration() 
{
    if (m_nu * m_dt / m_dx / m_dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * m_dx * m_dy / m_nu << endl;
        return LidDrivenCavityConfigError::FAILED;
    } else {
        return LidDrivenCavityConfigError::SUCCESS;
    }
}


/**
 * @brief Clean up all the dynamically assigned variables. Should
 * be called before being LidDrivenCavity object is destroyed.
 * 
 */
void LidDrivenCavity::CleanUp()
{
    if (m_v) {
        delete[] m_v;
        delete[] m_vnew;
        delete[] m_vtemp;
        delete[] m_s;
        delete[] m_tmp;
        delete m_cg;
    }
    if (m_localHeights) {
        delete[] m_localHeights;
        delete[] m_lengths;
        delete[] m_returnLengths;
        delete[] m_displacements;
        delete[] m_returnDisplacements;
        delete[] m_localArray1;
        delete[] m_localArray2;
        delete[] m_localArray3;
        delete[] m_localArray4;
    }
}


/**
 * @brief Update the terms associated with ∆x and ∆y.
 * 
 */
void LidDrivenCavity::UpdateDxDy()
{
    m_dx = m_Lx / (m_Nx - 1);
    m_dy = m_Ly / (m_Ny - 1);
    m_Npts = m_Nx * m_Ny;
    m_dxi = 1.0 / m_dx;
    m_dyi = 1.0 / m_dy;
    m_dx2i = 1.0 / m_dx / m_dx;
    m_dy2i = 1.0 / m_dy / m_dy;
    m_2dx2i = 2.0 * m_dx2i;
    m_2dy2i = 2.0 * m_dy2i;
}


/**
 * @brief Performs the time advacements step for the lid driven cavity
 * problem.
 * 
 */
void LidDrivenCavity::Advance()
{

    Vorticity(m_localArray1, m_localArray2);

    if (m_size > 1) {
        MPI_Sendrecv(&m_localArray2[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArray2[m_width * (m_localHeight + 1)], m_width, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArray2[m_localHeight * m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArray2, m_width, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    }

    TimeAdvance(m_localArray1, m_localArray2, m_localArray3);

    if (m_size > 1) {
        MPI_Sendrecv(&m_localArray3[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArray3[m_width*(m_localHeight+1)], m_width, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArray3[m_localHeight*m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArray3, m_width, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    }

    m_cg->Solve(m_localArray3, m_localArray1);
}


/**
 * @brief  Calculates vorticity boundary and interior conditions at time t:
 * 
 * @param    s                   Stream function matrix.
 * @param    v                   Vorticity matrix.
 */
void LidDrivenCavity::Vorticity(double *s, double *v)
{
    int i;
    int j;

    #pragma omp parallel shared(s, v, m_2dx2i, m_widthMinusOne, m_localHeightPlusOne) private(i)
    {
        # pragma omp sections nowait
        {
            #pragma omp section
            for (int i = 1; i < m_localHeightPlusOne; ++i) {
                v[IDX(i, 0)] = m_2dx2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
            }

            #pragma omp section
            for (int i = 1; i < m_localHeightPlusOne; ++i) {
                v[IDX(i, m_widthMinusOne)] = m_2dx2i * (s[IDX(i, m_widthMinusOne)] - s[IDX(i, m_widthMinusOne-1)]);
            }
        }
    }

    if (m_rank == 0) {
        for (int j = 1; j < m_widthMinusOne; ++j) {
            v[IDX(0, j)] = m_2dy2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }

    if (m_rank == m_size - 1) {
        for (int j = 1; j < m_widthMinusOne; ++j) {
            v[IDX(m_localHeightPlusOne, j)] = m_2dy2i * (s[IDX(m_localHeightPlusOne, j)] - s[IDX(m_localHeight, j)]) - 2.0 * m_dyi * m_U;
        }
    }

    // Compute interior vorticity
    # pragma omp parallel for shared(s, v, m_localHeightPlusOne, m_widthMinusOne, m_dy2i, m_dx2i) private(i, j)
    for (int i = 1; i < m_localHeightPlusOne; ++i) {
        for (int j = 1; j < m_widthMinusOne; ++j) {
            double term1 = 2.0 * s[IDX(i, j)];
            v[IDX(i, j)] = m_dy2i * (term1 - s[IDX(i + 1, j)] - s[IDX(i - 1, j)]) 
            + m_dx2i * (term1 - s[IDX(i, j + 1)] - s[IDX(i, j - 1)]);
        }
    }
}


/**
 * @brief Calculates the interior vorticity at time t + ∆t:
 * 
 * @param    s                   Stream function matrix.
 * @param    v                   Vorticity matrix.
 * @param    v_new               New vorticiy matrix.
 */
void LidDrivenCavity::TimeAdvance(double *s, double *v, double *v_new)
{
    int i;
    int j;
    # pragma omp parallel for shared(s, v, v_new, m_localHeightPlusOne, m_widthMinusOne, m_dyi, m_dxi, m_nu, m_dt) private(i, j)
    for (int i = 1; i < m_localHeightPlusOne; ++i) {
        for (int j = 1; j < m_widthMinusOne; ++j) {
            double term1 = v[IDX(i+1,j)];
            double term2 = v[IDX(i,j+1)];
            double term3 = v[IDX(i,j-1)];
            double term4 = v[IDX(i-1,j)];
            double term5 = 2.0*v[IDX(i,j)];

            // Calculate the gradients.
            double grad_x_s = (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * m_dxi;
            double grad_y_s = (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * m_dyi;
            double grad_y_v = (term1 - term4) * 0.5 * m_dyi;
            double grad_x_v = (term2 - term3) * 0.5 * m_dxi;

            // Calculate the advection terms.
            double advection_x = grad_x_s * grad_y_v;
            double advection_y = grad_y_s * grad_x_v;

            // Calculate the diffusion terms.
            double diffusion_x = m_nu * (term2 - term5 + term3)*m_dx2i;
            double diffusion_y = m_nu * (term1 - term5 + term4)*m_dy2i;

            // Combine all terms to get the final result.
            v_new[IDX(i, j)] = term5/2. + m_dt * (advection_x - advection_y + diffusion_x + diffusion_y);
        }
    }
}


/**
 * @brief Getter for the pointer to the vorticity data.
 * 
 * @return   const double* const Vorticity matrix.
 */
const double *const
LidDrivenCavity::GetVorticity()
    const
{
    return m_vnew;
}


/**
 * @brief Getter for the pointer to stream function data.
 * 
 * @return   const double* const Stream function matrix.
 */
const double *const
LidDrivenCavity::GetStreamFunction()
    const
{
    return m_s;
}


/**
 * @brief Getter for Nx.
 * 
 * @return   const double        Ny
 */
const double
LidDrivenCavity::GetNx()
    const
{
    return m_Nx;
}


/**
 * @brief Getter for Ny.
 * 
 * @return   const double        Ny.
 */
const double
LidDrivenCavity::GetNy()
    const
{
    return m_Ny;
}


/**
 * @brief Calculates all the necessary lengths, dimensions
 * and displacements needed so that the lid driven cavity
 * matrix can be separated into local matrices
 * which communicate with eachother using MPI.
 * 
 */
void LidDrivenCavity::SetSize()
{
    m_height = m_Ny;
    m_width = m_Nx;

    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);

    m_sizeX = std::sqrt(m_size);

    int smallHeightSize {(m_height-2) / m_size};
    int remainder {(m_height-2) % m_size};

    m_localHeights = new int[m_size]();
    m_lengths = new int[m_size]();
    m_returnLengths = new int[m_size]();
    m_displacements = new int[m_size]();
    m_returnDisplacements = new int[m_size]();

    int offset {};
    int returnOffset {};

    if (m_rank == 0) {
        m_start = 1;
    } else if (m_rank == m_size-1) {
        m_end = 1;
    }

    for (int i {}; i < m_size; ++i) {
        if (i < remainder) {
            m_localHeights[i] = smallHeightSize + 1;
        } else {
            m_localHeights[i] = smallHeightSize;
        }
        m_displacements[i] = offset;
        m_returnDisplacements[i] = returnOffset;
        m_lengths[i] = m_localHeights[i]*m_width + 2*m_width;

        if (i == 0) {
            m_returnLengths[i] += m_width;
            returnOffset += m_width;
        } 
        if (i == m_size -1) {
            m_returnLengths[i] += m_width;
        } 

        returnOffset += (m_localHeights[i])*m_width;
        m_returnLengths[i] += m_localHeights[i]*m_width;
        offset += (m_localHeights[i])*m_width;
    }

    m_localHeight = m_localHeights[m_rank];
    m_length = m_lengths[m_rank];
    m_returnLength = m_returnLengths[m_rank];

    m_localArray1 = new double[m_length]();
    m_localArray2 = new double[m_length]();
    m_localArray3 = new double[m_length]();
    m_localArray4 = new double[m_length]();

    m_left = m_rank - 1;
    m_right = m_rank + 1;
    if (m_start) {
        m_left = MPI_PROC_NULL;
    } else if (m_end) {
        m_right = MPI_PROC_NULL;
    }

    if (m_start) {
        m_returnStart = 0;
    } else {
        m_returnStart = m_width;
    }

    m_localHeightPlusOne = m_localHeight + 1;
    m_widthMinusOne = m_width - 1;
}
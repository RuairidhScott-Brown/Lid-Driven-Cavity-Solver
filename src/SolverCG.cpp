#include <iostream>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <cmath>
using namespace std;

#include <cblas.h>

#include "../include/SolverCG.h"

#define IDX(I,J) ((I)*m_Nx + (J))

/**
 * @brief Construct a new SolverCG:SolverCG object.
 * 
 * @param    pNx                 Number of grid points in x-direction.
 * @param    pNy                 Number of grid points in y-direction.
 * @param    pdx                 Delta x.
 * @param    pdy                 Delta y.
 * @param    comm                MPI communicator.
 */
SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy, MPI_Comm comm)
{
    m_dx = pdx;
    m_dy = pdy;
    m_Nx = pNx;
    m_Ny = pNy;
    m_height = pNy;
    m_width = pNx;

    m_dx2i = 1.0/m_dx/m_dx;
    m_dy2i = 1.0/m_dy/m_dy;
    m_factor = 2.0*(m_dx2i + m_dy2i);

    m_solver_comm = comm;

    UseMPI(true);

}


/**
 * @brief Sets the member variabes assocaited with indicating
 * the number of ranks available to the solver.
 * 
 * @param    yes                Flag to indicate if MPI should 
 *                              be utilised or not. Default is
 *                              true.
 */
void SolverCG::UseMPI(bool yes) {
    if (yes) {
        MPI_Comm_rank(m_solver_comm, &m_solver_rank);
        MPI_Comm_size(m_solver_comm, &m_solver_size);
    } else {
        m_solver_rank = 0;
        m_solver_size = 1;
    }
    SetSize();
    CreateMatrices();
}


/**
 * @brief Destroy the SolverCG::SolverCG object and cleans
 * up dynamically assigned memeory.
 * 
 */
SolverCG::~SolverCG()
{
    delete[] m_localArrayP;
    delete[] m_localArrayT;
    delete[] m_localArrayX;
    delete[] m_localArrayR;
    delete[] m_localArrayZ;
    delete[] m_localArrayB;
    delete[] m_localPre;
    delete[] m_localBC;
    delete[] m_localHeights;
    delete[] m_lengths;
    delete[] m_returnLengths;
    delete[] m_returnDisplacements;
    delete[] m_displacements;
}


/**
 * @brief Switch to direct the incoming data to an appropriate solver
 * depending on the number of processes active.
 * 
 * @param    b                   Matrix.
 * @param    x                   Matrix.
 * @return   SolverCGErrorCode   Enum class indicating the success of the solver.
 */
SolverCGErrorCode 
SolverCG::Solve(double* b, double* x) 
{
    if (m_solver_size > 1) {
        return SolveWithMultipleRank(b, x);
    } else {
        return SolveWithSingleRank(b, x);
    }
}


/**
 * @brief Main solver for the Poissions problem that utilises an interative scheme.
 * This function assumes that there is multiple MPI process runnning and as such
 * performs communication with the other ranks.
 * 
 * @param    b                   Matrix.
 * @param    x                   Matrix.
 * @return   SolverCGErrorCode   Enum class indicating the success of the solver.
 */
SolverCGErrorCode 
SolverCG::SolveWithMultipleRank(double* b, double* x) 
{
    m_k = 0;

    m_localArrayX = x;
    m_localArrayB = b;

    m_localDotProductEps = cblas_ddot(m_returnLength, &m_localArrayB[m_returnStart], 1, &m_localArrayB[m_returnStart], 1);
    MPI_Allreduce(&m_localDotProductEps, &m_globalDotProductEps, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

    m_eps = sqrt(m_globalDotProductEps);

    if (m_eps < m_tol) {
        std::memset(m_localArrayX, 0, m_length*sizeof(double));
        if (m_solver_rank == 0) {
            std::cout << "Norm is " << m_eps << std::endl;
        }
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::SUCCESS; // maybe another error code for this.
    }

    Laplace(m_localArrayX, m_localArrayT);


    MPI_Sendrecv(&m_localArrayT[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArrayT[m_width*(m_localHeight+1)], m_width, MPI_DOUBLE, m_right, 0, m_solver_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&m_localArrayT[m_localHeight*m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArrayT, m_width, MPI_DOUBLE, m_left, 0, m_solver_comm, MPI_STATUS_IGNORE);
    

    cblas_dcopy(m_length, m_localArrayB, 1, m_localArrayR, 1);
    cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_length, 0, m_localBC, 1, m_localArrayR, 1);

    cblas_daxpy(m_length, -1.0, m_localArrayT, 1, m_localArrayR, 1);
    cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
    cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayP, 1); 

    do {
        m_k++;
        
        Laplace(m_localArrayP, m_localArrayT);

        m_localDotProductAplha = cblas_ddot(m_returnLength, &m_localArrayT[m_returnStart], 1, &m_localArrayP[m_returnStart], 1);
        m_localDotProductBeta = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1);
        
        MPI_Allreduce(&m_localDotProductAplha, &m_alpha, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        MPI_Allreduce(&m_localDotProductBeta, &m_beta, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        m_alpha = m_beta / m_alpha;

        cblas_daxpy(m_length, m_alpha, m_localArrayP, 1, m_localArrayX, 1);
        cblas_daxpy(m_length, -m_alpha, m_localArrayT, 1, m_localArrayR, 1);

        m_localDotProductEps = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayR[m_returnStart], 1);
        MPI_Allreduce(&m_localDotProductEps, &m_globalDotProductEps, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

        m_eps = sqrt(m_globalDotProductEps);

        if (m_eps < m_tol) {
            break;
        }

        cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
        m_localDotProductTemp = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1);
        
        MPI_Allreduce(&m_localDotProductTemp, &m_globalDotProductTemp, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        m_beta = m_globalDotProductTemp / m_beta;

        cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayT, 1);
        cblas_daxpy(m_length, m_beta, m_localArrayP, 1, m_localArrayT, 1);
        cblas_dcopy(m_length, m_localArrayT, 1, m_localArrayP, 1);


        MPI_Sendrecv(&m_localArrayP[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArrayP[m_width*(m_localHeight+1)], m_width, MPI_DOUBLE, m_right, 0, m_solver_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArrayP[m_localHeight*m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArrayP, m_width, MPI_DOUBLE, m_left, 0, m_solver_comm, MPI_STATUS_IGNORE);


    } while (m_k < 5000); // Set a maximum number of iterations

    if (m_k == 5000) {
        std::cout << "FAILED TO CONVERGE" << std::endl;
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }
    m_localArrayX = nullptr;
    m_localArrayB = nullptr;

    if (m_solver_rank == 0) {
        std::cout << "Converged in " << m_k << " iterations. eps = " << m_eps << std::endl;
    }
    return SolverCGErrorCode::SUCCESS;
}


/**
 * @brief Main solver for the Poissions problem that utilises an interative scheme.
 * This function assumes that there is only one MPI process runnning and as such
 * does no communication with other ranks.
 * 
 * @param    b                   Matrix.
 * @param    x                   Matrix.
 * @return   SolverCGErrorCode   Enum class indicating the success of the solver.
 */
SolverCGErrorCode 
SolverCG::SolveWithSingleRank(double* b, double* x)
{
    m_k = 0;

    m_localArrayX = x;
    m_localArrayB = b;

    m_eps = sqrt(cblas_ddot(m_length, b, 1, b, 1));

    if (m_eps < m_tol) {
        std::memset(x, 0, m_length*sizeof(double));
        if (m_solver_rank == 0) {
            std::cout << "Norm is " << m_eps << std::endl;
        }
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::SUCCESS;
    }
    
    Laplace(x, m_localArrayT);

    cblas_dcopy(m_length, b, 1, m_localArrayR, 1);
    cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_length, 0, m_localBC, 1, m_localArrayR, 1);

    cblas_daxpy(m_length, -1.0, m_localArrayT, 1, m_localArrayR, 1);
    cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
    cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayP, 1); 

    do {
        m_k++;

        Laplace(m_localArrayP, m_localArrayT);

        #pragma omp parallel shared(m_localArrayP, m_localArrayT, m_localArrayR, m_localArrayZ, m_returnLength)
        {
            # pragma omp sections nowait
            {
                #pragma omp section
                m_alpha = cblas_ddot(m_returnLength, &m_localArrayT[m_returnStart], 1, &m_localArrayP[m_returnStart], 1);

                #pragma omp section
                m_beta = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1);
            }
        }
        m_alpha = m_beta / m_alpha;


        #pragma omp parallel shared(m_localArrayP, m_localArrayT, m_localArrayR, m_localArrayX, m_alpha)
        {
            # pragma omp sections nowait
            {
                #pragma omp section
                cblas_daxpy(m_length, m_alpha, m_localArrayP, 1, m_localArrayX, 1);

                #pragma omp section
                cblas_daxpy(m_length, -m_alpha, m_localArrayT, 1, m_localArrayR, 1);
            }
        }

        m_eps = sqrt(cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayR[m_returnStart], 1));

        if (m_eps < m_tol) {
            break;
        }

        cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
    
        m_beta = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1) / m_beta;

        cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayT, 1);
        cblas_daxpy(m_length, m_beta, m_localArrayP, 1, m_localArrayT, 1);
        cblas_dcopy(m_length, m_localArrayT, 1, m_localArrayP, 1);

    } while (m_k < 5000); // Set a maximum number of iterations
    if (m_k == 5000) {
        cout << "FAILED TO CONVERGE IN RANK: " << m_solver_rank <<  endl;
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }

    m_localArrayX = nullptr;
    m_localArrayB = nullptr;

    if (m_solver_rank == 0) {
        std::cout << "Converged in " << m_k << " iterations. eps = " << m_eps << std::endl;
    }
    return SolverCGErrorCode::SUCCESS;
}


/**
 * @brief Calculates all the necessary lengths, dimensions
 * and displacements needed so that the matrix data given to
 * SolverCG::Solver() can be separated into local matrices
 * which communicate with eachother using MPI.
 * 
 */
void SolverCG::SetSize()
{    
    int smallHeightSize {(m_height-2) / m_solver_size};
    int remainder {(m_height-2) % m_solver_size};

    m_localHeights = new int[m_solver_size]();
    m_lengths = new int[m_solver_size]();
    m_returnLengths = new int[m_solver_size]();
    m_displacements = new int[m_solver_size]();
    m_returnDisplacements = new int[m_solver_size]();

    int offset {};
    int returnOffset {};

    if (m_solver_rank == 0) {
        m_start = 1;
    } else if (m_solver_rank == m_solver_size-1) {
        m_end = 1;
    }

    for (int i {}; i < m_solver_size; ++i) {
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
        if (i == m_solver_size -1) {
            m_returnLengths[i] += m_width;
        } 

        returnOffset += (m_localHeights[i])*m_width;
        m_returnLengths[i] += m_localHeights[i]*m_width;
        offset += (m_localHeights[i])*m_width;
    }

    m_localHeight = m_localHeights[m_solver_rank];
    m_length = m_lengths[m_solver_rank];
    m_returnLength = m_returnLengths[m_solver_rank];

    m_localArrayP = new double[m_length]();
    m_localArrayT = new double[m_length]();
    m_localArrayX = new double[m_length]();
    m_localArrayR = new double[m_length]();
    m_localArrayZ = new double[m_length]();
    m_localArrayB = new double[m_length]();

    m_left = m_solver_rank - 1;
    m_right = m_solver_rank + 1;
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


/**
 * @brief Performs the laplace operation on the input matrix
 * and puts the results in the output matrix. The input matrix is
 * unmodified.
 * 
 * @param    in                  Input matrix to apply laplace operator.
 * @param    out                 Output matrix to store the results.
 */
void SolverCG::Laplace(double* in, double* out) {
    int i;
    int j;
    # pragma omp parallel for shared(m_localArrayP, m_localArrayT, m_dx2i, m_dy2i, m_localHeightPlusOne, m_widthMinusOne) private(i,j)
    for (int i = 1; i <m_localHeightPlusOne; ++i) {
        for (int j = 1; j < m_widthMinusOne; ++j) {
            double term1 = 2.0 * in[IDX(i, j)];
            double term2 = (-in[IDX(i, j - 1)] + term1 - in[IDX(i, j + 1)]) * m_dx2i;
            double term3 = (-in[IDX(i - 1, j)] + term1 - in[IDX(i + 1, j)]) * m_dy2i;
            out[IDX(i, j)] = term3 + term2;
        }
    }
}


/**
 * @brief Creates the matricies needed to perform the pre-condition
 * and boundary condition steps in the solver.
 * 
 */
void SolverCG::CreateMatrices() 
{
    if(m_localPre) {
        delete[] m_localPre;
    }
    if(m_localBC) {
        delete[] m_localBC;
    }

    m_localPre = new double[m_length];
    m_localBC = new double[m_length];

    // Populate the pre-condition matrix multiplier.
    for (int i {}; i < m_localHeight+2; ++i) {
        for (int j {}; j < m_width; ++j) {
            if (m_start && i == 0) {
                m_localPre[IDX(i, j)] = 1;
            } else if (m_end && i == m_localHeight + 1) {
                m_localPre[IDX(i, j)] = 1;
            } else if (j == 0 || j == m_width-1) {
                m_localPre[IDX(i, j)] = 1;
            } else {
                m_localPre[IDX(i, j)] = 1./m_factor;
            }   
        }
    }

    // Populate the boundary condition matrix multiplier.
    for (int i {}; i < m_localHeight+2; ++i) {
        for (int j {}; j < m_width; ++j) {
            if (m_start && i == 0) {
                m_localBC[IDX(i, j)] = 0;
            } else if (m_end && i == m_localHeight + 1) {
                m_localBC[IDX(i, j)] = 0;
            } else if (j == 0 || j == m_width-1) {
                m_localBC[IDX(i, j)] = 0;
            } else {
                m_localBC[IDX(i, j)] = 1;
            }   
        }
    }
}

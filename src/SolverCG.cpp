#include <iostream>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <cmath>
using namespace std;

#include <cblas.h>

#include "../include/SolverCG.h"

#define IDX(I,J) ((I)*m_Nx + (J))


void 
SolverCG::print_matrix_row_major(double* M, int r, int c) 
{
    std::cout << "[" << std::endl;
    for(int i {}; i<r; i++){
        for(int j {}; j<c; j++) {
            std::cout << M[i*c + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy, MPI_Comm comm)
{
    m_dx = pdx;
    m_dy = pdy;
    m_Nx = pNx;
    m_Ny = pNy;
    m_height = pNy;
    m_width = pNx;

    m_solver_comm = comm;

    UseMPI(true);

}

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


SolverCGErrorCode 
SolverCG::Solve(double* b, double* x) 
{
    if (m_solver_size > 1) {
        return SolveWithMultipleRank(b, x);
    } else {
        return SolveWithSingleRank(b, x);
    }
}

SolverCGErrorCode 
SolverCG::SolveWithMultipleRank(double* b, double* x) 
{
    int k {};
    double alpha {};
    double beta {};
    double eps {};
    double tol = 0.001;
    double localDotProductAplha {};
    double localDotProductBeta {};
    double globalDotProductEps {};
    double localDotProductEps {};
    double localDotProductTemp {};
    double globalDotProductTemp {};

    // MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayX, m_l, MPI_DOUBLE, 0, m_solver_comm);
    // MPI_Scatterv(b, m_ls, m_disp, MPI_DOUBLE, m_localArrayB, m_l, MPI_DOUBLE, 0, m_solver_comm);
    m_localArrayX = x;
    m_localArrayB = b;

    localDotProductEps = cblas_ddot(m_returnLength, &m_localArrayB[m_returnStart], 1, &m_localArrayB[m_returnStart], 1);
    MPI_Allreduce(&localDotProductEps, &globalDotProductEps, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

    eps = sqrt(globalDotProductEps);

    if (eps < tol*tol) {
        std::memset(m_localArrayX, 0, m_length*sizeof(double));
        if (m_solver_rank == 0) {
            std::cout << "Norm is " << eps << std::endl;
        }
        // MPI_Allgatherv(&m_localArrayX[m_rstart], m_rl, MPI_DOUBLE, x, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
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
        k++;
        
        Laplace(m_localArrayP, m_localArrayT);

        localDotProductAplha = cblas_ddot(m_returnLength, &m_localArrayT[m_returnStart], 1, &m_localArrayP[m_returnStart], 1);
        localDotProductBeta = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1);
        
        MPI_Allreduce(&localDotProductAplha, &alpha, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        MPI_Allreduce(&localDotProductBeta, &beta, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        alpha = beta / alpha;

        cblas_daxpy(m_length, alpha, m_localArrayP, 1, m_localArrayX, 1);
        cblas_daxpy(m_length, -alpha, m_localArrayT, 1, m_localArrayR, 1);

        localDotProductEps = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayR[m_returnStart], 1);
        MPI_Allreduce(&localDotProductEps, &globalDotProductEps, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

        eps = sqrt(globalDotProductEps);

        if (eps < tol*tol) {
            break;
        }

        cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
        localDotProductTemp = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1);
        
        MPI_Allreduce(&localDotProductTemp, &globalDotProductTemp, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        beta = globalDotProductTemp / beta;

        cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayT, 1);
        cblas_daxpy(m_length, beta, m_localArrayP, 1, m_localArrayT, 1);
        cblas_dcopy(m_length, m_localArrayT, 1, m_localArrayP, 1);


        MPI_Sendrecv(&m_localArrayP[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArrayP[m_width*(m_localHeight+1)], m_width, MPI_DOUBLE, m_right, 0, m_solver_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArrayP[m_localHeight*m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArrayP, m_width, MPI_DOUBLE, m_left, 0, m_solver_comm, MPI_STATUS_IGNORE);


    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        // std::cout << "FAILED TO CONVERGE" << std::endl;
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }
    m_k++;
    m_localArrayX = nullptr;
    m_localArrayB = nullptr;

    // MPI_Allgatherv(&m_localArrayX[m_rstart], m_rl, MPI_DOUBLE, x, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);

    // if (m_solver_rank == 0) {
    //     std::cout << "Converged in " << k << " iterations. eps = " << eps << std::endl;
    // }
    return SolverCGErrorCode::SUCCESS;
}

SolverCGErrorCode 
SolverCG::SolveWithSingleRank(double* b, double* x)
{
    int k {};
    double alpha {};
    double beta {};
    double eps {};
    double tol = 0.001;

    m_localArrayX = x;
    m_localArrayB = b;

    eps = sqrt(cblas_ddot(m_length, b, 1, b, 1));

    if (eps < tol*tol) {
        std::memset(x, 0, m_length*sizeof(double));
        if (m_solver_rank == 0) {
            std::cout << "Norm is " << eps << std::endl;
        }
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::SUCCESS; // maybe another error code for this.
    }
    
    // print_matrix_row_major(x, m_Ny, m_Nx);
    Laplace(x, m_localArrayT);
    // print_matrix_row_major(x, m_Ny, m_Nx);

    cblas_dcopy(m_length, b, 1, m_localArrayR, 1);
    cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_length, 0, m_localBC, 1, m_localArrayR, 1);
    // print_matrix_row_major(m_localArrayR, m_Ny, m_Nx);

    cblas_daxpy(m_length, -1.0, m_localArrayT, 1, m_localArrayR, 1);
    cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
    cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayP, 1); 
    // print_matrix_row_major(m_localArrayZ, m_Ny, m_Nx);
    do {
        k++;
        // if (k == 1) {
        //     print_matrix_row_major(m_localArrayT, m_Ny, m_Nx);
        // }
        Laplace(m_localArrayP, m_localArrayT);
        // if (k == 1) {
        //     print_matrix_row_major(m_localArrayT, m_Ny, m_Nx);
        //     std::cout << m_returnLength << std::endl;
        // }

        alpha = cblas_ddot(m_returnLength, &m_localArrayT[m_returnStart], 1, &m_localArrayP[m_returnStart], 1);
        // if (k == 1) {
        //     std::cout << alpha << std::endl;

        // }
        beta = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1);
        alpha = beta / alpha;
        // if (k == 1) {
        //     std::cout << alpha << std::endl;
        //     std::cout << beta << std::endl;

        // }

        cblas_daxpy(m_length, alpha, m_localArrayP, 1, m_localArrayX, 1);
        cblas_daxpy(m_length, -alpha, m_localArrayT, 1, m_localArrayR, 1);
        // if (k == 1) {
        //     print_matrix_row_major(m_localArrayR, m_Ny, m_Nx);
        // }

        eps = sqrt(cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayR[m_returnStart], 1));

        if (eps < tol*tol) {
            break;
        }

        cblas_dsbmv(CblasRowMajor, CblasUpper, m_length, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
    
        beta = cblas_ddot(m_returnLength, &m_localArrayR[m_returnStart], 1, &m_localArrayZ[m_returnStart], 1) / beta;

        cblas_dcopy(m_length, m_localArrayZ, 1, m_localArrayT, 1);
        cblas_daxpy(m_length, beta, m_localArrayP, 1, m_localArrayT, 1);
        cblas_dcopy(m_length, m_localArrayT, 1, m_localArrayP, 1);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        // cout << "FAILED TO CONVERGE IN RANK: " << m_solver_rank <<  endl;
        m_localArrayX = nullptr;
        m_localArrayB = nullptr;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }
    // print_matrix_row_major(m_localArrayX, m_Ny, m_Nx);
    // print_matrix_row_major(m_localArrayB, m_Ny, m_Nx);

    m_k++;
    m_localArrayX = nullptr;
    m_localArrayB = nullptr;
    // print_matrix_row_major(m_localArrayX, m_Ny, m_Nx);

    // MPI_Allgatherv(&m_localArrayX[m_rstart], m_rl, MPI_DOUBLE, x, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);

    // if (m_solver_rank == 0) {
    //     std::cout << "Converged in " << k << " iterations. eps = " << eps << std::endl;
    // }
    return SolverCGErrorCode::SUCCESS;
}



// double 
// SolverCG::MPI_cblas_ddot(const int m, const double* const x, const double* const y)
// {
//     if (!m_useMPI) {
//         return cblas_ddot(m, x, 1, y, 1);
//     }

//     // MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     // MPI_Scatterv(y, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(y, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

//     // double localDotProduct = cblas_ddot(m_localSize, m_localArray1, 1, m_localArray2, 1);
//     double localDotProduct = cblas_ddot(m_width*m_height, &m_localArrayP[m_height], 1, &m_localArrayT[m_height], 1);


//     double globalDotProduct;
//     MPI_Allreduce(&localDotProduct, &globalDotProduct, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

//     return globalDotProduct;
// }

 
// void SolverCG::MPI_cblas_daxpy(const int m, const double alpha, double* const x, double* const y)
// {
//     if (!m_useMPI) {
//         cblas_daxpy(m, alpha, x, 1, y, 1);
//         return;
//     }

//     // MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     // MPI_Scatterv(y, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(y, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

//     cblas_daxpy(m_width*m_height, alpha, &m_localArrayP[m_height], 1, &m_localArrayT[m_height], 1);
//     // cblas_daxpy(m_localSize, alpha, m_localArray1, 1, m_localArray2, 1);

//     // MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, y, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);

//     MPI_Allgatherv(&m_localArrayT[m_rstart], m_rl, MPI_DOUBLE, y, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
//     return ;
// }

// void SolverCG::MPI_cblas_dcopy(const int m, double* const x, double* const y)
// {
//     if (!m_useMPI) {
//         cblas_dcopy(m, x, 1, y, 1);
//         return;
//     }

//     MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(y, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);

//     cblas_dcopy(m_localSize, m_localArray1, 1, m_localArray2, 1);

//     MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, y, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);
//     return ;
// }

// double 
// SolverCG::MPI_cblas_dnrm2(const int m, const double* const x)
// {
//     if (!m_useMPI) {
//         return cblas_dnrm2(m, x, 1);
//     }
//     // MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

//     // double localDotProduct = cblas_ddot(m_localSize, m_localArray1, 1, m_localArray1, 1);
//     double localDotProduct = cblas_ddot(m_width*m_height, &m_localArrayT[m_height], 1, &m_localArrayT[m_height], 1);

//     double globalDotProduct;
//     MPI_Allreduce(&localDotProduct, &globalDotProduct, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

//     return sqrt(globalDotProduct);
// }

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

    m_localArrayP = new double[m_length](); // Local block of first vector
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
}

// void SolverCG::MPI_ApplyOperator(double* in, double* out) 
// {
//     if (!m_useMPI) {
//         ApplyOperator(in, out);
//         return;
//     }
//     cblas_dcopy(m_l, m_localArrayX, 1, m_localArrayP, 1);
//     cblas_dcopy(m_l, m_localArrayX, 1, m_localArrayT, 1);
//     MPI_Scatterv(in, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(out, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     Laplace(m_localArrayP, m_localArrayT);
//     MPI_Allgatherv(&(m_localArrayT[m_rstart]), m_rl, MPI_DOUBLE, out, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
//     return;
// } 


void SolverCG::Laplace(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    int jm1 = 0;
    int jp1 = 2;

    for (int i = 1; i < m_localHeight+1; ++i) {
        for (int j = 1; j < m_width-1; ++j) {
            out[IDX(i,j)] = ( -     in[IDX(i, j-1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, j+1)])*dx2i
                          + ( -     in[IDX(jm1, j)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(jp1, j)])*dy2i;
        }
        jm1++;
        jp1++;
    }
}

/**
 * @brief Applies the 2nd order laplacian to the input and
 * copies it across to the output.
 * 
 * @param    in     Matrix array input.
 * @param    out    Matrix array output.
 */
// void SolverCG::ApplyOperator(double* in, double* out) {
//     // Assume ordered with y-direction fastest (column-by-column)
//     double dx2i = 1.0/m_dx/m_dx;
//     double dy2i = 1.0/m_dy/m_dy;
//     int jm1 = (m_startNy !=0) ? 0 + m_startNy - 1 : 0;
//     int jp1 = (m_startNy !=0) ? 2 + m_startNy - 1 : 2;
//     double* data {};
//     if (m_useMPI) {
//         data = m_t2;
//     } else {
//         data = out;
//     }

//     for (int j = m_startNy; j < m_endNy; ++j) {
//         for (int i = m_startNx; i < m_endNx; ++i) {
//             data[IDX(i,j)] = ( -    in[IDX(i-1, j)]
//                               + 2.0*in[IDX(i,   j)]
//                               -     in[IDX(i+1, j)])*dx2i
//                           + ( -     in[IDX(i, jm1)]
//                               + 2.0*in[IDX(i,   j)]
//                               -     in[IDX(i, jp1)])*dy2i;
//         }
//         jm1++;
//         jp1++;
//     }
//     if (m_useMPI) {
//         MPI_Allreduce(m_t2, out, m_Nx*m_Ny, MPI_DOUBLE, MPI_SUM, m_solver_comm);
//         std::memset(m_t2, 0, m_Nx*m_Ny*sizeof(double));
//     }
// }

// void SolverCG::PopulatePreconditionBandedMatrix()
// {
//     double dx2i = 1.0/m_dx/m_dx;
//     double dy2i = 1.0/m_dy/m_dy;
//     double factor = 2.0*(dx2i + dy2i);

//     for (int j {}; j < m_Ny; ++j) {
//         for (int i {}; i < m_Nx; ++i) {
//             if (j == 0 || j == (m_Ny-1)) {
//                 m_pre[IDX(i,j)] = 1;
//             } else if (i == 0 || i == (m_Nx-1)) {
//                 m_pre[IDX(i,j)] = 1;
//             } else {
//                 m_pre[IDX(i,j)] = 1./factor;
//             }
//         }
//     }
// }

// void SolverCG::PopulateImposeBCBandedMatrix()
// {
//     for (int j {}; j < m_Ny; ++j) {
//         for (int i {}; i < m_Nx; ++i) {
//             if (j == 0 || j == (m_Ny-1)) {
//                 m_bc[IDX(i,j)] = 0;
//             } else if (i == 0 || i == (m_Nx-1)) {
//                 m_bc[IDX(i,j)] = 0;
//             } else {
//                 m_bc[IDX(i,j)] = 1;
//             }
//         }
//     }
// }

/**
 * @brief Calaculates the the expression Mij / 2(dx + dy) for all
 * interior points of the the input matrix and copies that
 * expression across to the output matrix. The boundaries remain
 * the same.
 * 
 * @param    in     Matrix array input.
 * @param    out    Matrix array output.
 */
// void SolverCG::MPI_Precondition(double* in, double* out)
// {   if (!m_useMPI) {
//         cblas_dsbmv(CblasColMajor, CblasUpper, m_Nx*m_Ny, 0, 1.0, m_pre, 1, in, 1, 0.0, out, 1);
//         return;
//     }

//     // MPI_Scatterv(in, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     // MPI_Scatterv(out, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(in, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(out, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

//     // cblas_dsbmv(CblasColMajor, CblasUpper, m_localSize, 0, 1.0, m_localPre, 1, m_localArray1, 1, 0.0, m_localArray2, 1);
//     cblas_dsbmv(CblasColMajor, CblasUpper, m_l, 0, 1.0, m_localPre, 1, m_localArrayP, 1, 0.0, m_localArrayT, 1);

//     // MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, out, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);
//     MPI_Allgatherv(&m_localArrayT[m_rstart], m_rl, MPI_DOUBLE, out, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
//     return ;
// }

// void SolverCG::MPI_ImposeBC(double* out)
// {   if (!m_useMPI) {
//         cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_Nx*m_Ny, 0, m_bc, 1, out, 1);
//         return;
//     }

//     MPI_Scatterv(out, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);

//     cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_localSize, 0, m_localBC, 1, m_localArray1, 1);

//     MPI_Allgatherv(m_localArray1, m_localSize, MPI_DOUBLE, out, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);
//     return ;
// }

// void SolverCG::SetCommunicator() 
// {
//     m_useMPI = true;
//     MPI_Comm_rank(MPI_COMM_WORLD, &m_globalRank);
//     MPI_Comm_size(MPI_COMM_WORLD, &m_size);

//     MPI_Group world_group;
//     MPI_Comm_group(MPI_COMM_WORLD, &world_group);

//     int n = m_size;
//     if (n > (m_Nx-2)) {
//         n = m_Nx-2;
//     }

//     int* ranks  = new int[n];
//     for (int i {}; i < n; ++i) {
//         ranks[i] = i;
//     }

//     MPI_Group prime_group;
//     MPI_Group_incl(world_group, n, ranks, &prime_group);

//     MPI_Comm_create(MPI_COMM_WORLD, prime_group, &m_solver_comm);

//     if (MPI_COMM_NULL != m_solver_comm) {
//         MPI_Comm_rank(m_solver_comm, &m_solver_rank);
//         MPI_Comm_size(m_solver_comm, &m_solver_size);
//     }

//     std::cout << "World: " << m_globalRank << "/" << m_size << "Prime: " << m_solver_rank << "/" << m_solver_size << std::endl;
//     delete[] ranks;

//     SetSize(m_size);
//     CreateMatrices();

//     MPI_Scatterv(m_p, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(m_t, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(m_r, m_ls, m_disp, MPI_DOUBLE, m_localArrayR, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(m_z, m_ls, m_disp, MPI_DOUBLE, m_localArrayZ, m_l, MPI_DOUBLE, 0, m_solver_comm);
// }

void SolverCG::CreateMatrices() 
{
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    double factor = 2.0*(dx2i + dy2i);

    if(m_localPre) {
        delete[] m_localPre;
    }
    if(m_localBC) {
        delete[] m_localBC;
    }

    m_localPre = new double[m_length];
    m_localBC = new double[m_length];

    for (int i {}; i < m_localHeight+2; ++i) {
        for (int j {}; j < m_width; ++j) {
            if (m_start && i == 0) {
                m_localPre[IDX(i, j)] = 1;
            } else if (m_end && i == m_localHeight + 1) {
                m_localPre[IDX(i, j)] = 1;
            } else if (j == 0 || j == m_width-1) {
                m_localPre[IDX(i, j)] = 1;
            } else {
                m_localPre[IDX(i, j)] = 1./factor;
            }   
        }
    }

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

#include <iostream>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <cmath>
using namespace std;

#include <cblas.h>

#include "../include/SolverCG.h"

#define IDX(I,J) ((J)*m_Nx + (I))

SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    m_dx = pdx;
    m_dy = pdy;
    m_Nx = pNx;
    m_Ny = pNy;
    int n = m_Nx*m_Ny;
    m_r = new double[n];
    m_p = new double[n];
    m_z = new double[n];
    m_t = new double[n]; //temp
    m_t2 = new double[n]; //temp
}


SolverCG::~SolverCG()
{
    delete[] m_r;
    delete[] m_p;
    delete[] m_z;
    delete[] m_t;
    delete[] m_t2;
    if (m_arrays) {
        delete[] m_arrays;
        delete[] m_disp;
    }
    if (m_localArray1) {
        delete[] m_localArray1;
        delete[] m_localArray2;
    }
}


SolverCGErrorCode 
SolverCG::Solve(double* b, double* x) 
{
    unsigned int n = m_Nx*m_Ny;
    int k {};
    double alpha;
    double beta;
    double eps;
    double tol = 0.001;

    eps = MPI_cblas_dnrm2(n, b);
    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return SolverCGErrorCode::SUCCESS; // maybe another error code for this.
    }

    ApplyOperator(x, m_t);
    // cblas_dcopy(n, b, 1, m_r, 1);        // r_0 = b (i.e. b)
    MPI_cblas_dcopy(n, b, m_r);        // r_0 = b (i.e. b)
    ImposeBC(m_r);

    MPI_cblas_daxpy(n, -1.0, m_t, m_r);
    Precondition(m_r, m_z);
    // cblas_dcopy(n, m_z, 1, m_p, 1);        // p_0 = r_0
    MPI_cblas_dcopy(n, m_z, m_p);        // p_0 = r_0


    do {
        k++;
        // Perform action of Nabla^2 * m_p
        ApplyOperator(m_p, m_t);

        alpha = MPI_cblas_ddot(n, m_t, m_p);  // alpha = p_k^T A p_k
        beta  = MPI_cblas_ddot(n, m_r, m_z);  // z_k^T r_k
        alpha = beta / alpha;

        MPI_cblas_daxpy(n, alpha, m_p, x);  // x_{k+1} = x_k + alpha_k p_k
        MPI_cblas_daxpy(n, -alpha, m_t, m_r); // r_{k+1} = r_k - alpha_k A p_k


        eps = MPI_cblas_dnrm2(n, m_r);

        if (eps < tol*tol) {
            break;
        }
        Precondition(m_r, m_z);
        beta = MPI_cblas_ddot(n, m_r, m_z) / beta;

        // cblas_dcopy(n, m_z, 1, m_t, 1);
        MPI_cblas_dcopy(n, m_z, m_t);
        MPI_cblas_daxpy(n, beta, m_p, m_t);
        MPI_cblas_dcopy(n, m_t, m_p);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        // cout << "FAILED TO CONVERGE" << endl;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }

    // if (m_rankCol == 0 && m_rankRow == 0) {
    //     cout << "Converged in " << k << " iterations. eps = " << eps << endl;
    // }
    return SolverCGErrorCode::SUCCESS;
}


double 
SolverCG::MPI_cblas_ddot(const int m, const double* const x, const double* const y)
{
    if (!m_useMPI) {
        return cblas_ddot(m, x, 1, y, 1);
    }

    MPI_Scatterv(x, m_arrays, m_disp, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_grid);
    MPI_Scatterv(y, m_arrays, m_disp, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_grid);

    double localDotProduct = cblas_ddot(m_localSize, m_localArray1, 1, m_localArray2, 1);

    double globalDotProduct;
    MPI_Allreduce(&localDotProduct, &globalDotProduct, 1, MPI_DOUBLE, MPI_SUM, m_grid);

    return globalDotProduct;
}

 
void SolverCG::MPI_cblas_daxpy(const int m, const double alpha, double* const x, double* const y)
{
    if (!m_useMPI) {
        cblas_daxpy(m, alpha, x, 1, y, 1);
        return;
    }

    MPI_Scatterv(x, m_arrays, m_disp, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_grid);
    MPI_Scatterv(y, m_arrays, m_disp, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_grid);

    cblas_daxpy(m_localSize, alpha, m_localArray1, 1, m_localArray2, 1);

    MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, y, m_arrays, m_disp, MPI_DOUBLE, m_grid);
    return ;
}

void SolverCG::MPI_cblas_dcopy(const int m, double* const x, double* const y)
{
    if (!m_useMPI) {
        cblas_dcopy(m, x, 1, y, 1);
        return;
    }

    MPI_Scatterv(x, m_arrays, m_disp, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_grid);
    MPI_Scatterv(y, m_arrays, m_disp, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_grid);

    cblas_dcopy(m_localSize, m_localArray1, 1, m_localArray2, 1);

    MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, y, m_arrays, m_disp, MPI_DOUBLE, m_grid);
    return ;
}

double 
SolverCG::MPI_cblas_dnrm2(const int m, const double* const x)
{
    if (!m_useMPI) {
        return cblas_dnrm2(m, x, 1);
    }
    MPI_Scatterv(x, m_arrays, m_disp, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_grid);

    double localDotProduct = cblas_ddot(m_localSize, m_localArray1, 1, m_localArray1, 1);

    double globalDotProduct;
    MPI_Allreduce(&localDotProduct, &globalDotProduct, 1, MPI_DOUBLE, MPI_SUM, m_grid);

    return sqrt(globalDotProduct);
}

/**
 * @brief Applies the 2nd order laplacian to the input and
 * copies it across to the output.
 * 
 * @param    in     Matrix array input.
 * @param    out    Matrix array output.
 */
void SolverCG::ApplyOperator(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    int jm1 = (m_startNy !=0) ? 0 + m_startNy - 1 : 0;
    int jp1 = (m_startNy !=0) ? 2 + m_startNy - 1 : 2;
    double* data {};
    if (m_useMPI) {
        data = m_t2;
    } else {
        data = out;
    }

    for (int j = m_startNy; j < m_endNy; ++j) {
        for (int i = m_startNx; i < m_endNx; ++i) {
            data[IDX(i,j)] = ( -     in[IDX(i-1, j)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i+1, j)])*dx2i
                          + ( -     in[IDX(i, jm1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, jp1)])*dy2i;
        }
        jm1++;
        jp1++;
    }
    if (m_useMPI) {
        MPI_Allreduce(m_t2, out, m_Nx*m_Ny, MPI_DOUBLE, MPI_SUM, m_grid);
        std::memset(m_t2, 0, m_Nx*m_Ny*sizeof(double));
    }
}

/**
 * @brief Calaculates the the expression Mij / 2(dx + dy) for all
 * interior points of the the input matrix and copies that
 * expression across to the output matrix. The boundaries remain
 * the same.
 * 
 * @param    in     Matrix array input.
 * @param    out    Matrix array output.
 */
void SolverCG::Precondition(double* in, double* out) {
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    double factor = 2.0*(dx2i + dy2i);
    double* data {};
    if (m_useMPI) {
        data = m_t2;
    } else {
        data = out;
    }

    for (int i = m_startNx; i < m_endNx; ++i) {
        for (int j = m_startNy; j < m_endNy; ++j) {
            data[IDX(i,j)] = in[IDX(i,j)]/factor;
        }
    }

    if (m_rankRow == 0) {
        for (int i = m_startNx-1; i < m_endNx+1; ++i) {
            data[IDX(i, 0)] = in[IDX(i,0)];
        }
    }
    if (m_rankRow == (m_sizeX-1)) {
        for (int i = m_startNx-1; i < m_endNx+1; ++i) {
            data[IDX(i, m_Ny-1)] = in[IDX(i, m_Ny-1)];
        }
    }

    if (m_rankCol == 0) {
        for (int j = m_startNy-1; j < m_endNy+1; ++j) {
            data[IDX(0, j)] = in[IDX(0, j)];
        }
    }

    if (m_rankCol == (m_sizeX-1)) {
        for (int j = m_startNy-1; j < m_endNy+1; ++j) {
            data[IDX(m_Nx - 1, j)] = in[IDX(m_Nx - 1, j)];
        }
    }

    if (m_useMPI) {
        MPI_Allreduce(m_t2, out, m_Nx*m_Ny, MPI_DOUBLE, MPI_SUM, m_grid);
        std::memset(m_t2, 0, m_Nx*m_Ny*sizeof(double));
    }
}

/**
 * @brief Modifies the input matrix such that all
 * the boundaries have a value of 0.
 * 
 * @param    inout  Matrix array to be modified.
 */
void SolverCG::ImposeBC(double* inout) { //will this be faster with MPI??
        // Boundaries
    for (int i = 0; i < m_Nx; ++i) {
        inout[IDX(i, 0)] = 0.0;
        inout[IDX(i, m_Ny-1)] = 0.0;
    }

    for (int j = 0; j < m_Ny; ++j) {
        inout[IDX(0, j)] = 0.0;
        inout[IDX(m_Nx - 1, j)] = 0.0;
    }
}

void SolverCG::SetCommunicator(MPI_Comm grid) 
{
    m_grid = grid;
    m_useMPI = true;
    MPI_Comm_rank(grid, &m_globalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
    SetSize(m_size);
}

void SolverCG::SetRank(int rankRow, int rankCol)
{
    m_rankRow = rankRow;
    m_rankCol = rankCol;
}

void SolverCG::SetSize(int size)
{
    m_sizeX = sqrt(m_size);
    int n {m_Nx*m_Ny};
    int blockSize {n / m_size};
    int remainder {n % m_size};
    m_arrays = new int[m_size];
    m_disp = new int[m_size];
    int offset {};


    for (int i {}; i < m_size; ++i) {
        if (i < remainder) {
            m_arrays[i] = blockSize + 1;
        } else {
            m_arrays[i] = blockSize;
        }
        m_disp[i] = offset;
        offset += m_arrays[i];
    }

    m_localSize = m_arrays[m_globalRank];
    m_localArray1 = new double[m_localSize]; // Local block of first vector
    m_localArray2 = new double[m_localSize];
}

void SolverCG::SetSubGridDimensions(int startNx, int endNx, int startNy, int endNy) 
{
    m_startNx = startNx;
    m_startNy = startNy;
    m_endNx = endNx;
    m_endNy = endNy;
}

void SolverCG::CalculateSubGridDimensions() {
    if (!m_useMPI) {
        m_sizeX = 1;
        m_size = 1;
    }
    int lNx {m_Nx / m_sizeX};
    int lNy {m_Ny / m_sizeX};

    int lrNx {m_Nx % m_sizeX};
    int lrNy {m_Ny % m_sizeX};

    int lastX {};
    int secondLastX {};
    int thirdLastX {};

    int lastY {};
    int secondLastY {};
    int thirdLastY {};

    if (m_sizeX > 1 && m_rankCol == m_sizeX-1) lastX = 1;
    if (m_sizeX >= 3 && m_rankCol >= m_sizeX-2) secondLastX = 1;
    if (m_sizeX >= 4 && m_rankCol >= m_sizeX-3) thirdLastX = 1;

    if (m_sizeX > 1 && m_rankRow == m_sizeX-1) lastY = 1;
    if (m_sizeX >= 3 && m_rankRow >= m_sizeX-2) secondLastY = 1;
    if (m_sizeX >= 4 && m_rankRow >= m_sizeX-3) thirdLastY = 1;


    switch (lrNx) {
        case 0:
            m_startNx = m_rankCol*lNx;
            m_endNx = m_startNx + lNx-1;
            break;
        case 1:
            m_startNx = m_rankCol*lNx;
            m_endNx = m_startNx + lNx-1 + lastX;
            break;
        case 2:
            m_startNx = m_rankCol*lNx + lastX;
            m_endNx = m_startNx + lNx-1 + secondLastX;
            break;
        case 3:
            m_startNx = m_rankCol*lNx + secondLastX + lastX;
            m_endNx = m_startNx + lNx-1 + thirdLastX;
            break;
    }

    switch (lrNy) {
        case 0:
            m_startNy = m_rankRow*lNy;
            m_endNy = m_startNy + lNy-1;
            break;
        case 1:
            m_startNy = m_rankRow*lNy;
            m_endNy = m_startNy + lNy-1 + lastY;
            break;
        case 2:
            m_startNy = m_rankRow*lNy + lastY;
            m_endNy = m_startNy + lNy-1 + secondLastY;
            break;
        case 3:
            m_startNy = m_rankCol*lNy + secondLastY + lastY;
            m_endNy = m_startNy + lNy-1 + thirdLastY;
            break;
    }

    m_endNx++;
    m_endNy++;

    if (m_rankCol == 0) m_startNx++;
    if (m_rankCol == (m_sizeX-1)) m_endNx--;

    if (m_rankRow == 0) m_startNy++;
    if (m_rankRow == (m_sizeX-1)) m_endNy--;
}

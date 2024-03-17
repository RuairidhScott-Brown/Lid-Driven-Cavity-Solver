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
}


SolverCGErrorCode 
SolverCG::Solve(double* b, double* x) 
{
    unsigned int n = m_Nx*m_Ny;
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.001;

    eps = cblas_dnrm2(n, b, 1);
    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return SolverCGErrorCode::SUCCESS; // maybe another error code for this.
    }

    ApplyOperator(x, m_t);
    cblas_dcopy(n, b, 1, m_r, 1);        // r_0 = b (i.e. b)
    ImposeBC(m_r);

    cblas_daxpy(n, -1.0, m_t, 1, m_r, 1);
    Precondition(m_r, m_z);
    cblas_dcopy(n, m_z, 1, m_p, 1);        // p_0 = r_0

    k = 0;
    do {
        k++;
        // Perform action of Nabla^2 * m_p
        ApplyOperator(m_p, m_t);

        alpha = cblas_ddot(n, m_t, 1, m_p, 1);  // alpha = p_k^T A p_k
        alpha = cblas_ddot(n, m_r, 1, m_z, 1) / alpha; // compute alpha_k
        beta  = cblas_ddot(n, m_r, 1, m_z, 1);  // z_k^T r_k

        cblas_daxpy(n,  alpha, m_p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, m_t, 1, m_r, 1); // r_{k+1} = r_k - alpha_k A p_k

        eps = cblas_dnrm2(n, m_r, 1);

        if (eps < tol*tol) {
            break;
        }
        Precondition(m_r, m_z);
        beta = cblas_ddot(n, m_r, 1, m_z, 1) / beta;

        cblas_dcopy(n, m_z, 1, m_t, 1);
        cblas_daxpy(n, beta, m_p, 1, m_t, 1);
        cblas_dcopy(n, m_t, 1, m_p, 1);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        // cout << "FAILED TO CONVERGE" << endl;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }

    // cout << "Converged in " << k << " iterations. eps = " << eps << endl;
    return SolverCGErrorCode::SUCCESS;
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
void SolverCG::ImposeBC(double* inout) {
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
}

void SolverCG::SetRank(int rankRow, int rankCol)
{
    m_rankRow = rankRow;
    m_rankCol = rankCol;
}

void SolverCG::SetSize(int size)
{
    m_size = size;
    m_sizeX = sqrt(m_size);
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

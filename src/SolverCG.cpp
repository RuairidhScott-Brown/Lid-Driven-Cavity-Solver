#include <iostream>
#include <algorithm>
#include <cstring>
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
}


SolverCG::~SolverCG()
{
    delete[] m_r;
    delete[] m_p;
    delete[] m_z;
    delete[] m_t;
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
        return SolverCGErrorCode::SUCCESS;
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
        cout << "FAILED TO CONVERGE" << endl;
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
    int jm1 = 0;
    int jp1 = 2;
    for (int j = 1; j < m_Ny - 1; ++j) {
        for (int i = 1; i < m_Nx - 1; ++i) {
            out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i+1, j)])*dx2i
                          + ( -     in[IDX(i, jm1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, jp1)])*dy2i;
        }
        jm1++;
        jp1++;
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
    int i;
    int j;
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    double factor = 2.0*(dx2i + dy2i);
    for (i = 1; i < m_Nx - 1; ++i) {
        for (j = 1; j < m_Ny - 1; ++j) {
            out[IDX(i,j)] = in[IDX(i,j)]/factor;
        }
    }
    // Boundaries
    for (i = 0; i < m_Nx; ++i) {
        out[IDX(i, 0)] = in[IDX(i,0)];
        out[IDX(i, m_Ny-1)] = in[IDX(i, m_Ny-1)];
    }

    for (j = 0; j < m_Ny; ++j) {
        out[IDX(0, j)] = in[IDX(0, j)];
        out[IDX(m_Nx - 1, j)] = in[IDX(m_Nx - 1, j)];
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

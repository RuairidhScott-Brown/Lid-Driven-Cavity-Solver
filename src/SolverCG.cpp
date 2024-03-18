#include <iostream>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <cmath>
using namespace std;

#include <cblas.h>

#include "../include/SolverCG.h"

#define IDX(I,J) ((J)*m_Ny + (I))

SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    m_dx = pdx;
    m_dy = pdy;
    m_Nx = pNx;
    m_Ny = pNy;
    m_height = pNy;
    int n = m_Nx*m_Ny;
    m_r = new double[n];
    m_p = new double[n];
    m_z = new double[n];
    m_t = new double[n]; //temp
    m_t2 = new double[n]; //temp
    m_pre = new double[n];
    m_bc = new double[n];
    PopulatePreconditionBandedMatrix();
    PopulateImposeBCBandedMatrix();

}


SolverCG::~SolverCG()
{
    delete[] m_r;
    delete[] m_p;
    delete[] m_z;
    delete[] m_t;
    delete[] m_t2;
    delete[] m_pre;
    delete[] m_bc;
    if (m_localArray1) {
        delete[] m_localArray1;
        delete[] m_localArray2;
        delete[] m_localArrayP;
        delete[] m_localArrayT;
        delete[] m_localArrayX;
        delete[] m_localArrayR;
        delete[] m_localArrayZ;
        delete[] m_localPre;
        delete[] m_localBC;
        delete[] m_widths;
        delete[] m_ls;
        delete[] m_rls;
        delete[] m_rdisp;
        delete[] m_disp;
        delete[] m_disp2;
        delete[] m_arrays;
    }
    // if (m_local) {
    //     delete[] m_local;
    // }
}


SolverCGErrorCode 
SolverCG::Solve(double* b, double* x) 
{
    unsigned int n = m_Nx*m_Ny;
    int k {};
    m_alpha = 0;
    m_beta = 0;
    // double alpha {};
    // double beta {};
    // double eps;
    m_eps = 0;
    double tol = 0.001;

    m_eps = MPI_cblas_dnrm2(n, b);
    if (m_eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << m_eps << endl;
        return SolverCGErrorCode::SUCCESS; // maybe another error code for this.
    }


    MPI_ApplyOperator(x, m_t);

    MPI_cblas_dcopy(n, b, m_r);        // r_0 = b (i.e. b)
    MPI_ImposeBC(m_r);

    MPI_cblas_daxpy(n, -1.0, m_t, m_r);
    MPI_Precondition(m_r, m_z);
    MPI_cblas_dcopy(n, m_z, m_p);        // p_0 = r_0

    MPI_Scatterv(m_p, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);

    MPI_Scatterv(m_t, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(m_r, m_ls, m_disp, MPI_DOUBLE, m_localArrayR, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(m_z, m_ls, m_disp, MPI_DOUBLE, m_localArrayZ, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayX, m_l, MPI_DOUBLE, 0, m_solver_comm);

    do {
        k++;
        // Perform action of Nabla^2 * m_p
        // MPI_ApplyOperator(m_p, m_t);

        // cblas_dscal(m_l, 0.0, m_localArrayP, 1);
        // cblas_dscal(m_l, 0.0, m_localArrayT, 1);
        // cblas_dscal(m_l, 0.0, m_localArrayX, 1);
        // cblas_dscal(m_l, 0.0, m_localArrayR, 1);
        // cblas_dscal(m_l, 0.0, m_localArrayZ, 1);
        // MPI_Scatterv(m_p, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
        // MPI_Scatterv(m_t, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);
        // MPI_Scatterv(m_r, m_ls, m_disp, MPI_DOUBLE, m_localArrayR, m_l, MPI_DOUBLE, 0, m_solver_comm);
        // MPI_Scatterv(m_z, m_ls, m_disp, MPI_DOUBLE, m_localArrayZ, m_l, MPI_DOUBLE, 0, m_solver_comm);
        // MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayX, m_l, MPI_DOUBLE, 0, m_solver_comm);
        
        // MPI_ApplyOperator(m_p, m_t);
        Laplace(m_localArrayP, m_localArrayT);

        // m_alpha = MPI_cblas_ddot(m_Nx*m_Ny, m_t, m_p);  // alpha = p_k^T A p_k
        // beta  = MPI_cblas_ddot(n, m_r, m_z);  // z_k^T r_k
        // alpha = beta / alpha;
        double localDotProductAplha = cblas_ddot(m_width*m_height, &m_localArrayT[m_height], 1, &m_localArrayP[m_height], 1);  // alpha = p_k^T A p_k
        double localDotProductBeta = cblas_ddot(m_width*m_height, &m_localArrayR[m_height], 1, &m_localArrayZ[m_height], 1);  // alpha = p_k^T A p_k
        
        MPI_Allreduce(&localDotProductAplha, &m_alpha, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        MPI_Allreduce(&localDotProductBeta, &m_beta, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        m_alpha = m_beta / m_alpha;

        // MPI_cblas_daxpy(m_width*m_height, m_alpha, m_p, x);  // x_{k+1} = x_k + alpha_k p_k
        // MPI_cblas_daxpy(m_width*m_height, -m_alpha, m_t, m_r); // r_{k+1} = r_k - alpha_k A p_k
        // cblas_daxpy(m_width*m_height, m_alpha, &m_localArrayP[m_height], 1, &m_localArrayX[m_height], 1);
        cblas_daxpy(m_l, m_alpha, m_localArrayP, 1, m_localArrayX, 1);
        cblas_daxpy(m_l, -m_alpha, m_localArrayT, 1, m_localArrayR, 1);

        // eps = MPI_cblas_dnrm2(n, m_r);
        double localDotProductEps = cblas_ddot(m_width*m_height, &m_localArrayR[m_height], 1, &m_localArrayR[m_height], 1);
        double globalDotProductEps;
        MPI_Allreduce(&localDotProductEps, &globalDotProductEps, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

        m_eps = sqrt(globalDotProductEps);


        // MPI_Allgatherv(&m_localArrayX[m_rstart], m_rl, MPI_DOUBLE, x, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayR[m_rstart], m_rl, MPI_DOUBLE, m_r, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayT[m_rstart], m_rl, MPI_DOUBLE, m_t, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayP[m_rstart], m_rl, MPI_DOUBLE, m_p, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayZ[m_rstart], m_rl, MPI_DOUBLE, m_z, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);

        // m_alpha = MPI_cblas_ddot(n, m_t, m_p);  // alpha = p_k^T A p_k
        // m_beta  = MPI_cblas_ddot(n, m_r, m_z);  // z_k^T r_k
        // m_alpha = m_beta / m_alpha;

        // MPI_cblas_daxpy(n, m_alpha, m_p, x);  // x_{k+1} = x_k + alpha_k p_k
        // MPI_cblas_daxpy(n, -m_alpha, m_t, m_r); // r_{k+1} = r_k - alpha_k A p_k


        // m_eps = MPI_cblas_dnrm2(n, m_r);

        if (m_eps < tol*tol) {
            break;
        }

        // MPI_Precondition(m_r, m_z);
        cblas_dsbmv(CblasColMajor, CblasUpper, m_l, 0, 1.0, m_localPre, 1, m_localArrayR, 1, 0.0, m_localArrayZ, 1);
        MPI_Allgatherv(&m_localArrayZ[m_rstart], m_rl, MPI_DOUBLE, m_z, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // m_beta = MPI_cblas_ddot(n, m_r, m_z) / m_beta;
        double localDotProductTemp = cblas_ddot(m_width*m_height, &m_localArrayR[m_height], 1, &m_localArrayZ[m_height], 1);
        
        double globalDotProductTemp {};
        MPI_Allreduce(&localDotProductTemp, &globalDotProductTemp, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        m_beta = globalDotProductTemp / m_beta;

        // MPI_cblas_dcopy(n, m_z, m_t);
        // cblas_dcopy(n, m_z, 1, m_t, 1);
        cblas_dcopy(m_l, m_localArrayZ, 1, m_localArrayT, 1);

        // MPI_cblas_daxpy(n, m_beta, m_p, m_t); //you could prbable use the tragular opeator there
        cblas_daxpy(m_l, m_beta, m_localArrayP, 1, m_localArrayT, 1);

        // MPI_cblas_dcopy(n, m_t, m_p);
        cblas_dcopy(m_l, m_localArrayT, 1, m_localArrayP, 1);


        if (m_solver_size > 1) {
            MPI_Sendrecv(&m_localArrayP[m_height], m_height, MPI_DOUBLE, m_left, 0, &m_localArrayP[m_height*(m_width+1)], m_height, MPI_DOUBLE, m_right, 0, m_solver_comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&m_localArrayP[m_height*m_width], m_height, MPI_DOUBLE, m_right, 0, m_localArrayP, m_height, MPI_DOUBLE, m_left, 0, m_solver_comm, MPI_STATUS_IGNORE);
        }
        // MPI_Allgatherv(&m_localArrayR[m_rstart], m_rl, MPI_DOUBLE, m_r, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayT[m_rstart], m_rl, MPI_DOUBLE, m_t, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayP[m_rstart], m_rl, MPI_DOUBLE, m_p, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
        // MPI_Allgatherv(&m_localArrayZ[m_rstart], m_rl, MPI_DOUBLE, m_z, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        // cout << "FAILED TO CONVERGE" << endl;
        return SolverCGErrorCode::CONVERGE_FAILED;
    }
    m_k++;

    MPI_Allgatherv(&m_localArrayX[m_rstart], m_rl, MPI_DOUBLE, x, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);

    // if (m_rankCol == 0 && m_rankRow == 0) {
    //     cout << "Converged in " << k << " iterations. eps = " << eps << endl;
    // }
    return SolverCGErrorCode::SUCCESS;
}


// void SolverCG::temp(double* m_p, double* m_t, double* m_r, double* m_z, double* x) 
// {
//     // Perform action of Nabla^2 * m_p
//     // MPI_ApplyOperator(m_p, m_t);
//     cblas_dscal(m_l, 0.0, m_localArray11, 1);
//     cblas_dscal(m_l, 0.0, m_localArray22, 1);
//     cblas_dscal(m_l, 0.0, m_localArray33, 1);
//     cblas_dscal(m_l, 0.0, m_localArray44, 1);
//     cblas_dscal(m_l, 0.0, m_localArray55, 1);
//     MPI_Scatterv(m_p, m_ls, m_disp, MPI_DOUBLE, m_localArray11, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(m_t, m_ls, m_disp, MPI_DOUBLE, m_localArray22, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(m_r, m_ls, m_disp, MPI_DOUBLE, m_localArray44, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(m_z, m_ls, m_disp, MPI_DOUBLE, m_localArray55, m_l, MPI_DOUBLE, 0, m_solver_comm);
//     MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArray33, m_l, MPI_DOUBLE, 0, m_solver_comm);
    
//     // MPI_ApplyOperator(m_p, m_t);
//     Laplace(m_localArray11, m_localArray22);

//     // m_alpha = MPI_cblas_ddot(m_Nx*m_Ny, m_t, m_p);  // alpha = p_k^T A p_k
//     // beta  = MPI_cblas_ddot(n, m_r, m_z);  // z_k^T r_k
//     // alpha = beta / alpha;
//     double localDotProductAplha = cblas_ddot(m_width*m_height, &m_localArray22[m_height], 1, &m_localArray11[m_height], 1);  // alpha = p_k^T A p_k
//     double localDotProductBeta = cblas_ddot(m_width*m_height, &m_localArray44[m_height], 1, &m_localArray55[m_height], 1);  // alpha = p_k^T A p_k
    
//     MPI_Allreduce(&localDotProductAplha, &m_alpha, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
//     MPI_Allreduce(&localDotProductBeta, &m_beta, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);
//     m_alpha = m_beta / m_alpha;

//     // MPI_cblas_daxpy(m_width*m_height, m_alpha, m_p, x);  // x_{k+1} = x_k + alpha_k p_k
//     // MPI_cblas_daxpy(m_width*m_height, -m_alpha, m_t, m_r); // r_{k+1} = r_k - alpha_k A p_k
//     cblas_daxpy(m_width*m_height, m_alpha, &m_localArray11[m_height], 1, &m_localArray33[m_height], 1);
//     cblas_daxpy(m_width*m_height, -m_alpha, &m_localArray22[m_height], 1, &m_localArray44[m_height], 1);

//     // eps = MPI_cblas_dnrm2(n, m_r);
//     double localDotProductEps = cblas_ddot(m_width*m_height, &m_localArray44[m_height], 1, &m_localArray44[m_height], 1);
//     double globalDotProductEps;
//     MPI_Allreduce(&localDotProductEps, &globalDotProductEps, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

//     m_eps = sqrt(globalDotProductEps);


//     MPI_Allgatherv(&m_localArray33[m_rstart], m_rl, MPI_DOUBLE, x, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
//     MPI_Allgatherv(&m_localArray44[m_rstart], m_rl, MPI_DOUBLE, m_r, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);

//     MPI_Allgatherv(&(m_localArray22[m_rstart]), m_rl, MPI_DOUBLE, m_t, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
// }


double 
SolverCG::MPI_cblas_ddot(const int m, const double* const x, const double* const y)
{
    if (!m_useMPI) {
        return cblas_ddot(m, x, 1, y, 1);
    }

    // MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    // MPI_Scatterv(y, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(y, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

    // double localDotProduct = cblas_ddot(m_localSize, m_localArray1, 1, m_localArray2, 1);
    double localDotProduct = cblas_ddot(m_width*m_height, &m_localArrayP[m_height], 1, &m_localArrayT[m_height], 1);


    double globalDotProduct;
    MPI_Allreduce(&localDotProduct, &globalDotProduct, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

    return globalDotProduct;
}

 
void SolverCG::MPI_cblas_daxpy(const int m, const double alpha, double* const x, double* const y)
{
    if (!m_useMPI) {
        cblas_daxpy(m, alpha, x, 1, y, 1);
        return;
    }

    // MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    // MPI_Scatterv(y, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(y, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

    cblas_daxpy(m_width*m_height, alpha, &m_localArrayP[m_height], 1, &m_localArrayT[m_height], 1);
    // cblas_daxpy(m_localSize, alpha, m_localArray1, 1, m_localArray2, 1);

    // MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, y, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);

    MPI_Allgatherv(&m_localArrayT[m_rstart], m_rl, MPI_DOUBLE, y, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
    return ;
}

void SolverCG::MPI_cblas_dcopy(const int m, double* const x, double* const y)
{
    if (!m_useMPI) {
        cblas_dcopy(m, x, 1, y, 1);
        return;
    }

    MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(y, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);

    cblas_dcopy(m_localSize, m_localArray1, 1, m_localArray2, 1);

    MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, y, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);
    return ;
}

double 
SolverCG::MPI_cblas_dnrm2(const int m, const double* const x)
{
    if (!m_useMPI) {
        return cblas_dnrm2(m, x, 1);
    }
    // MPI_Scatterv(x, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(x, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

    // double localDotProduct = cblas_ddot(m_localSize, m_localArray1, 1, m_localArray1, 1);
    double localDotProduct = cblas_ddot(m_width*m_height, &m_localArrayT[m_height], 1, &m_localArrayT[m_height], 1);

    double globalDotProduct;
    MPI_Allreduce(&localDotProduct, &globalDotProduct, 1, MPI_DOUBLE, MPI_SUM, m_solver_comm);

    return sqrt(globalDotProduct);
}

void SolverCG::SetSize(int size)
{
    m_sizeX = sqrt(m_size);
    
    int smallWidthSize {(m_height-2) / m_solver_size};
    int remainder {(m_height-2) % m_solver_size};

    m_widths = new int[m_solver_size];
    m_ls = new int[m_solver_size];
    m_rls = new int[m_solver_size];
    m_disp = new int[m_solver_size];
    m_rdisp = new int[m_solver_size];

    int offset {};
    int roffset {};

    if (m_solver_rank == 0) {
        m_start = 1;
    } else if (m_solver_rank == m_solver_size-1) {
        m_end = 1;
    }

    for (int i {}; i < m_solver_size; ++i) {
        if (i < remainder) {
            m_widths[i] = smallWidthSize + 1;
        } else {
            m_widths[i] = smallWidthSize;
        }
        m_disp[i] = offset;
        m_rdisp[i] = roffset;
        m_ls[i] = m_widths[i]*m_height + 2*m_height;

        if (i == 0) {
            m_rls[i] = m_widths[i]*m_height + m_height;
            roffset += (m_widths[i])*m_height + m_height;
        } else if (i == m_solver_size -1) {
            m_rls[i] = m_widths[i]*m_height + m_height;
            roffset += (m_widths[i])*m_height;
        } else {
            m_rls[i] = m_widths[i]*m_height;
            roffset += (m_widths[i])*m_height;

        }
        offset += (m_widths[i])*m_height;
    }

    m_width = m_widths[m_solver_rank];
    m_l = m_width*m_height + 2*m_height;
    m_rl = m_rls[m_solver_rank];

    m_localArrayP = new double[m_l](); // Local block of first vector
    m_localArrayT = new double[m_l]();
    m_localArrayX = new double[m_l]();
    m_localArrayR = new double[m_l]();
    m_localArrayZ = new double[m_l]();

    m_left = m_solver_rank - 1;
    m_right = m_solver_rank + 1;
    if (m_start) {
        m_left = MPI_PROC_NULL;
    } else if (m_end) {
        m_right = MPI_PROC_NULL;
    }

    if (m_start) {
        m_rstart = 0;
    } else {
        m_rstart = m_height;
    }

    int n {m_Nx*m_Ny};
    int blockSize {n / m_solver_size};
    remainder = n % m_solver_size;
    m_arrays = new int[m_solver_size];
    m_disp2 = new int[m_solver_size];
    int offset2 {};


    for (int i {}; i < m_solver_size; ++i) {
        if (i < remainder) {
            m_arrays[i] = blockSize + 1;
        } else {
            m_arrays[i] = blockSize;
        }
        m_disp2[i] = offset2;
        offset2 += m_arrays[i];
    }

    m_localSize = m_arrays[m_solver_rank];
    m_localArray1 = new double[m_localSize]; // Local block of first vector
    m_localArray2 = new double[m_localSize];
}

void SolverCG::MPI_ApplyOperator(double* in, double* out) 
{
    if (!m_useMPI) {
        ApplyOperator(in, out);
        return;
    }
    cblas_dcopy(m_l, m_localArrayX, 1, m_localArrayP, 1);
    cblas_dcopy(m_l, m_localArrayX, 1, m_localArrayT, 1);
    MPI_Scatterv(in, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(out, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);
    Laplace(m_localArrayP, m_localArrayT);
    MPI_Allgatherv(&(m_localArrayT[m_rstart]), m_rl, MPI_DOUBLE, out, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
    return;
} 


void SolverCG::Laplace(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    int jm1 = 0;
    int jp1 = 2;

    for (int j = 1; j < m_width+1; ++j) {
        for (int i = 1; i < m_height-1; ++i) {
            out[IDX(i,j)] = ( -    in[IDX(i-1, j)]
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
            data[IDX(i,j)] = ( -    in[IDX(i-1, j)]
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
        MPI_Allreduce(m_t2, out, m_Nx*m_Ny, MPI_DOUBLE, MPI_SUM, m_solver_comm);
        std::memset(m_t2, 0, m_Nx*m_Ny*sizeof(double));
    }
}

void SolverCG::PopulatePreconditionBandedMatrix()
{
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    double factor = 2.0*(dx2i + dy2i);

    for (int j {}; j < m_Ny; ++j) {
        for (int i {}; i < m_Nx; ++i) {
            if (j == 0 || j == (m_Ny-1)) {
                m_pre[IDX(i,j)] = 1;
            } else if (i == 0 || i == (m_Nx-1)) {
                m_pre[IDX(i,j)] = 1;
            } else {
                m_pre[IDX(i,j)] = 1./factor;
            }
        }
    }
}

void SolverCG::PopulateImposeBCBandedMatrix()
{
    for (int j {}; j < m_Ny; ++j) {
        for (int i {}; i < m_Nx; ++i) {
            if (j == 0 || j == (m_Ny-1)) {
                m_bc[IDX(i,j)] = 0;
            } else if (i == 0 || i == (m_Nx-1)) {
                m_bc[IDX(i,j)] = 0;
            } else {
                m_bc[IDX(i,j)] = 1;
            }
        }
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
void SolverCG::MPI_Precondition(double* in, double* out)
{   if (!m_useMPI) {
        cblas_dsbmv(CblasColMajor, CblasUpper, m_Nx*m_Ny, 0, 1.0, m_pre, 1, in, 1, 0.0, out, 1);
        return;
    }

    // MPI_Scatterv(in, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    // MPI_Scatterv(out, m_arrays, m_disp2, MPI_DOUBLE, m_localArray2, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(in, m_ls, m_disp, MPI_DOUBLE, m_localArrayP, m_l, MPI_DOUBLE, 0, m_solver_comm);
    MPI_Scatterv(out, m_ls, m_disp, MPI_DOUBLE, m_localArrayT, m_l, MPI_DOUBLE, 0, m_solver_comm);

    // cblas_dsbmv(CblasColMajor, CblasUpper, m_localSize, 0, 1.0, m_localPre, 1, m_localArray1, 1, 0.0, m_localArray2, 1);
    cblas_dsbmv(CblasColMajor, CblasUpper, m_l, 0, 1.0, m_localPre, 1, m_localArrayP, 1, 0.0, m_localArrayT, 1);

    // MPI_Allgatherv(m_localArray2, m_localSize, MPI_DOUBLE, out, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);
    MPI_Allgatherv(&m_localArrayT[m_rstart], m_rl, MPI_DOUBLE, out, m_rls, m_rdisp, MPI_DOUBLE, m_solver_comm);
    return ;
}

void SolverCG::MPI_ImposeBC(double* out)
{   if (!m_useMPI) {
        cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_Nx*m_Ny, 0, m_bc, 1, out, 1);
        return;
    }

    MPI_Scatterv(out, m_arrays, m_disp2, MPI_DOUBLE, m_localArray1, m_localSize, MPI_DOUBLE, 0, m_solver_comm);

    cblas_dtbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, m_localSize, 0, m_localBC, 1, m_localArray1, 1);

    MPI_Allgatherv(m_localArray1, m_localSize, MPI_DOUBLE, out, m_arrays, m_disp2, MPI_DOUBLE, m_solver_comm);
    return ;
}

void SolverCG::SetCommunicator(MPI_Comm grid) 
{
    m_grid = grid;
    m_useMPI = true;
    MPI_Comm_rank(grid, &m_globalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    int n = m_size;
    if (n > (m_Nx-2)) {
        n = m_Nx-2;
    }

    int* ranks  = new int[n];
    for (int i {}; i < n; ++i) {
        ranks[i] = i;
    }

    MPI_Group prime_group;
    MPI_Group_incl(world_group, n, ranks, &prime_group);

    MPI_Comm_create(MPI_COMM_WORLD, prime_group, &m_solver_comm);

    if (MPI_COMM_NULL != m_solver_comm) {
        MPI_Comm_rank(m_solver_comm, &m_solver_rank);
        MPI_Comm_size(m_solver_comm, &m_solver_size);
    }

    std::cout << "World: " << m_globalRank << "/" << m_size << "Prime: " << m_solver_rank << "/" << m_solver_size << std::endl;
    delete[] ranks;

    SetSize(m_size);
    CreateMatrices();
}

void SolverCG::CreateMatrices() 
{
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;
    double factor = 2.0*(dx2i + dy2i);

    // int n {m_width*m_height};
    if(m_localPre) {
        delete[] m_localPre;
    }
    if(m_localBC) {
        delete[] m_localBC;
    }
    m_localPre = new double[m_l];
    // m_localPre = new double[m_localSize];
    m_localBC = new double[m_localSize];
    // MPI_Scatterv(m_pre, m_arrays, m_disp2, MPI_DOUBLE, m_localPre, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
    for (int j {}; j < m_width+2; ++j) {
        for (int i {}; i < m_height; ++i) {
            if (m_start && j == 0) {
                m_localPre[IDX(i, j)] = 1;
            } else if (m_end && j == m_width + 1) {
                m_localPre[IDX(i, j)] = 1;
            } else if (i == 0 || i == m_height-1) {
                m_localPre[IDX(i, j)] = 1;
            } else {
                m_localPre[IDX(i, j)] = 1./factor;
            }   
        }
    }
    // if (m_solver_rank == 3) {
    //     for (int i {}; i < m_l; ++i) {
    //         std::cout << m_localPre2[i] << " ";
    //     }
    //     std::cout << std::endl; 
    // } 
    MPI_Scatterv(m_bc, m_arrays, m_disp2, MPI_DOUBLE, m_localBC, m_localSize, MPI_DOUBLE, 0, m_solver_comm);
}

void SolverCG::SetRank(int rankRow, int rankCol)
{
    m_rankRow = rankRow;
    m_rankCol = rankCol;
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

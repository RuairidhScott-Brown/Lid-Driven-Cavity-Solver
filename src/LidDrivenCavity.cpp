#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>

// #define IDX(I, J) ((J)*m_Ny + (I))
#define IDX(I,J) ((I)*m_Nx + (J))

#include "../include/LidDrivenCavity.h"
#include "../include/SolverCG.h"


void 
print_matrix_row_major(double* M, int r, int c) 
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


void print_vector(const double *V, int l)
{
    for (int i{}; i < l; i++)
    {
        std::cout << V[i] << " ";
    }
    std::cout << std::endl;
}


LidDrivenCavity::LidDrivenCavity(MPI_Comm comm)
{
    m_comm = comm;
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    m_Lx = xlen;
    m_Ly = ylen;
    UpdateDxDy();
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    m_Nx = nx;
    m_Ny = ny;
    UpdateDxDy();
}

void LidDrivenCavity::SetTimeStep(double deltat)
{
    m_dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
    m_T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{
    m_Re = re;
    m_nu = 1.0 / re;
}

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

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(m_T / m_dt);

    for (int t = 0; t < NSteps; ++t) {
        if (m_rank == 0) {
        //     std::cout << "Step: " << setw(8) << t
        //             << "  Time: " << setw(8) << t*m_dt
        //             << std::endl;
        }
        Advance();
        m_k++;
    }
    MPI_Allgatherv(&m_localArray3[m_returnStart], m_returnLength, MPI_DOUBLE, m_vnew, m_returnLengths, m_returnDisplacements, MPI_DOUBLE, m_comm);
    MPI_Allgatherv(&m_localArray1[m_returnStart], m_returnLength, MPI_DOUBLE, m_s, m_returnLengths, m_returnDisplacements, MPI_DOUBLE, m_comm);
    // if (m_rank == 0) {
    //     std::cout << "After" << std::endl;
    //     print_matrix_row(m_s, 9);
    //     // print_vector(m_localArray1, m_l);
    // }
}

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

void LidDrivenCavity::ConvertStreamFunctionToVelocityV(double *const v)
{
    for (int j = 1; j < m_Nx - 1; ++j) {
        for (int i = 1; i < m_Ny - 1; ++i) {
            v[IDX(i, j)] = -(m_s[IDX(i, j+1)] - m_s[IDX(i, j)]) / m_dx;
        }
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    std::cout << std::fixed << std::setprecision(6);

    if (m_rankRow != 0 || m_rankCol != 0)
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

void LidDrivenCavity::PrintConfiguration()
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
    if (m_nu * m_dt / m_dx / m_dy > 0.25)
    {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * m_dx * m_dy / m_nu << endl;
        exit(-1);
    }
}

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

void LidDrivenCavity::UpdateDxDy()
{
    m_dx = m_Lx / (m_Nx - 1);
    m_dy = m_Ly / (m_Ny - 1);
    m_Npts = m_Nx * m_Ny;
}

void LidDrivenCavity::Advance()
{

    // std::memset(m_v, 0, m_Nx*m_Ny*sizeof(double));
    // std::memset(m_vnew, 0, m_Nx*m_Ny*sizeof(double));
    // std::memset(m_vtemp, 0, m_Nx*m_Ny*sizeof(double));

    // MPI_Scatterv(m_s, m_ls, m_disp, MPI_DOUBLE, m_localArray1, m_l, MPI_DOUBLE, 0, m_comm);

    // if (m_k == 51 && m_rank == 3) {
    //     std::cout << "After2" << std::endl;
    //     // print_matrix_row(m_s, 9);
    //     print_vector(m_localArray1, m_l);
    // }
    // MPI_Scatterv(m_v, m_ls, m_disp, MPI_DOUBLE, m_localArray2, m_l, MPI_DOUBLE, 0, m_comm);
    // MPI_Scatterv(m_vnew, m_ls, m_disp, MPI_DOUBLE, m_localArray3, m_l, MPI_DOUBLE, 0, m_comm);

    V(m_localArray1, m_localArray2);

    // if (m_k == 0) {
    //     print_matrix_row_major(m_localArray2, m_Ny, m_Nx);
    // }
    // // MPI_Allgatherv(&m_localArray3[m_rstart], m_rl, MPI_DOUBLE, m_vnew, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // // MPI_Allgatherv(&m_localArray2[m_rstart], m_rl, MPI_DOUBLE, m_v, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // // MPI_Scatterv(m_v, m_ls, m_disp, MPI_DOUBLE, m_localArray2, m_l, MPI_DOUBLE, 0, m_comm);
    // // if (m_k == 0 && m_rank == 3) {
    // //     std::cout << "before" << std::endl;
    // //     print_vector(m_localArray2, m_l);
    // // }

    // // MPI_Allgatherv(&m_localArray2[m_rstart], m_rl, MPI_DOUBLE, m_v, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // // MPI_Scatterv(m_v, m_ls, m_disp, MPI_DOUBLE, m_localArray2, m_l, MPI_DOUBLE, 0, m_comm);
    if (m_size > 1) {
        MPI_Sendrecv(&m_localArray2[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArray2[m_width * (m_localHeight + 1)], m_width, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArray2[m_localHeight * m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArray2, m_width, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    }

    // if (m_k == 1 && m_rank == 0) {
    //     std::cout << "before" << std::endl;
    //     print_matrix_row(m_v, 9);
    // }

    // V(m_s, m_v);

    // if (m_k == 1 && m_rank == 0) {
    //     std::cout << "After" << std::endl;
    //     print_matrix_row(m_v, 9);
    // }
    // MPI_TimeAdvance(m_s, m_v, m_vnew);

    // if (m_k == 0) {
    //     print_matrix_row_major(m_localArray2, m_Ny, m_Nx);
    // }
    TimeAdvance(m_localArray1, m_localArray2, m_localArray3);
    // if (m_k == 0) {
    //     print_matrix_row_major(m_localArray1, m_Ny, m_Nx);
    // }
    // MPI_Allgatherv(&m_localArray3[m_rstart], m_rl, MPI_DOUBLE, m_vnew, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // MPI_Allgatherv(&m_localArray1[m_rstart], m_rl, MPI_DOUBLE, m_s, m_rls, m_rdisp, MPI_DOUBLE, m_comm);

    if (m_size > 1) {
        MPI_Sendrecv(&m_localArray3[m_width], m_width, MPI_DOUBLE, m_left, 0, &m_localArray3[m_width*(m_localHeight+1)], m_width, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArray3[m_localHeight*m_width], m_width, MPI_DOUBLE, m_right, 0, m_localArray3, m_width, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    }
    // // Solve Poisson problem
    // m_cg->Solve(m_vnew, m_localArray1);

    // if (m_k == 0) {
    //     print_matrix_row_major(m_localArray3, m_Ny, m_Nx);
    // }
    m_cg->Solve(m_localArray3, m_localArray1);
    // print_matrix_row_major(m_localArray1, m_Ny, m_Nx);

    // if (m_k == 0) {
    //     print_matrix_row_major(m_localArray1, m_Ny, m_Nx);
    // }
    // if (m_size > 1) {
    //     MPI_Sendrecv(&m_localArray2[m_height], m_height, MPI_DOUBLE, m_left, 0, &m_localArray2[m_height*(m_width+1)], m_height, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
    //     MPI_Sendrecv(&m_localArray2[m_height*m_width], m_height, MPI_DOUBLE, m_right, 0, m_localArray2, m_height, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    // }
    // // m_cg->Solve(m_localArray3, m_localArray1);
    // MPI_Allgatherv(&m_localArray3[m_rstart], m_rl, MPI_DOUBLE, m_vnew, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // MPI_Allgatherv(&m_localArray1[m_rstart], m_rl, MPI_DOUBLE, m_s, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // MPI_Scatterv(m_s, m_ls, m_disp, MPI_DOUBLE, m_localArray4, m_l, MPI_DOUBLE, 0, m_comm);


    // if (m_rank == 3) {
    //     // std::cout << "After" << std::endl;
    //     // print_matrix_row(m_tmp, 9);
    //     print_vector(m_localArray1, m_l);
    // }
    //     if (m_k == 50 && m_rank == 3) {
    //     std::cout << "After2" << std::endl;
    //     // print_matrix_row(m_s, 9);
    //     print_vector(m_localArray1, m_l);
    // }
    // MPI_V(m_s, m_v);
    // MPI_TimeAdvance(m_s, m_v, m_vnew);
    // if (m_size > 1) {
    //     MPI_Sendrecv(&m_localArray1[m_height], m_height, MPI_DOUBLE, m_left, 0, &m_localArray1[m_height*(m_width+1)], m_height, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
    //     MPI_Sendrecv(&m_localArray1[m_height*m_width], m_height, MPI_DOUBLE, m_right, 0, m_localArray1, m_height, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    // }

    // Solve Poisson problem
    // m_cg->Solve(m_vnew, m_s);
}

// void LidDrivenCavity::MPI_V(double *s, double *v)
// {
//     MPI_Scatterv(s, m_ls, m_disp, MPI_DOUBLE, m_localArray1, m_l, MPI_DOUBLE, 0, m_comm);
//     MPI_Scatterv(v, m_ls, m_disp, MPI_DOUBLE, m_localArray2, m_l, MPI_DOUBLE, 0, m_comm);
//     V(m_localArray1, m_localArray2);
//     // if (m_rank == 0 && m_k ==1) {
//     //     print_vector(m_localArray2, m_l);
//     // }
//     MPI_Allgatherv(&m_localArray2[m_rstart], m_rl, MPI_DOUBLE, v, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
// }

void LidDrivenCavity::V(double *s, double *v)
{
    double dyi = 1.0 / m_dy;
    double dx2i = 1.0 / m_dx / m_dx;
    double dy2i = 1.0 / m_dy / m_dy;

    for (int i = 1; i < m_localHeight + 1; ++i) {
        v[IDX(i, 0)] = 2.0 * dx2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        v[IDX(i, m_width - 1)] = 2.0 * dx2i * (s[IDX(i, m_width-1)] - s[IDX(i, m_width-2)]);
    }

    if (m_rank == 0) {
        for (int j = 1; j < m_width - 1; ++j) {
            v[IDX(0, j)] = 2.0 * dy2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        }
    }

    if (m_rank == m_size - 1) {
        for (int j = 1; j < m_width - 1; ++j) {
            // if (m_k == 0) {
            //     std::cout << 2.0 * dy2i * (s[IDX(i, m_width + 1)] - s[IDX(i, m_width)]) - 2.0 * dyi * m_U << std::endl;
            // }
            v[IDX(m_localHeight + 1, j)] = 2.0 * dy2i * (s[IDX(m_localHeight + 1, j)] - s[IDX(m_localHeight, j)]) - 2.0 * dyi * m_U;
            if (m_k == 0) {
            //     std::cout << 2.0 * dy2i * (s[IDX(i, m_width + 1)] - s[IDX(i, m_width)]) - 2.0 * dyi * m_U << std::endl;
            // std::cout << v[IDX(i, m_width + 1)] << std::endl;
            }
        }
    }
    // if (m_k == 0) {
    //     std::cout << v[IDX(1, m_width + 1)] << std::endl;
    // }
    // if (m_k == 0) {
    //     print_matrix_row_major(v, m_Ny, m_Nx);
    // }
    // Compute interior vorticity
    for (int i = 1; i < m_localHeight + 1; ++i) {
        for (int j = 1; j < m_width - 1; ++j) {
            v[IDX(i, j)] = dy2i * (2.0 * s[IDX(i, j)] - s[IDX(i + 1, j)] - s[IDX(i - 1, j)]) 
            + 1.0 / m_dx / m_dx * (2.0 * s[IDX(i, j)] - s[IDX(i, j + 1)] - s[IDX(i, j - 1)]);
        }
    }
    // if (m_k == 0) {
    //     print_matrix_row_major(v, m_Ny, m_Nx);
    // }
}

// void LidDrivenCavity::MPI_TimeAdvance(double *s, double *v, double *v_new)
// {
//     // m_s, m_v, m_vtemp
//     // if (!m_useMPI) {
//     //     ApplyOperator(in, out);
//     //     return;
//     // }
//     // cblas_dcopy(m_l, m_localArray1, 1, m_localArrayP, 1);
//     // cblas_dcopy(m_l, m_localArrayX, 1, m_localArrayT, 1);
//     MPI_Scatterv(s, m_ls, m_disp, MPI_DOUBLE, m_localArray1, m_l, MPI_DOUBLE, 0, m_comm);
//     MPI_Scatterv(v, m_ls, m_disp, MPI_DOUBLE, m_localArray2, m_l, MPI_DOUBLE, 0, m_comm);
//     MPI_Scatterv(v_new, m_ls, m_disp, MPI_DOUBLE, m_localArray3, m_l, MPI_DOUBLE, 0, m_comm);
//     TimeAdvance(m_localArray1, m_localArray2, m_localArray3);
//     MPI_Allgatherv(&m_localArray3[m_rstart], m_rl, MPI_DOUBLE, v_new, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
//     return;
// }

void LidDrivenCavity::TimeAdvance(double *s, double *v, double *v_new)
{
    // m_s, m_v, m_vtemp
    double dxi = 1.0 / m_dx;
    double dyi = 1.0 / m_dy;
    double dx2i = 1.0 / m_dx / m_dx;
    double dy2i = 1.0 / m_dy / m_dy;

    // int ii {7};
    // int jj {1};
    // int test = v[IDX(ii,jj)] + m_dt*(
    //             ( (s[IDX(ii,jj+1)] - s[IDX(ii,jj-1)]) * 0.5 * dxi
    //              *(v[IDX(ii+1,jj)] - v[IDX(ii-1,jj)]) * 0.5 * dyi)
    //           - ( (s[IDX(ii+1,jj)] - s[IDX(ii-1,jj)]) * 0.5 * dyi
    //              *(v[IDX(ii,jj+1)] - v[IDX(ii,jj-1)]) * 0.5 * dxi)
    //           + m_nu * (v[IDX(ii,jj+1)] - 2.0 * v[IDX(ii,jj)] + v[IDX(ii,jj-1)])*dx2i
    //           + m_nu * (v[IDX(ii+1,jj)] - 2.0 * v[IDX(ii,jj)] + v[IDX(ii-1,jj)])*dy2i);

    // if (m_k == 0) {
    //     std::cout << m_localHeight << std::endl;
    //     std::cout << m_width << std::endl;
    //     std::cout << dxi << std::endl;
    //     std::cout << dyi << std::endl;
    //     std::cout << dx2i << std::endl;
    //     std::cout << dy2i << std::endl;
    //     std::cout << m_nu << std::endl;

    // }
    for (int i = 1; i < m_localHeight + 1; ++i) {
        for (int j = 1; j < m_width - 1; ++j) {
            v_new[IDX(i, j)] = v[IDX(i,j)] + m_dt*(
                ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dxi
                 *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dyi)
              - ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dyi
                 *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dxi)
              + m_nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dx2i
              + m_nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dy2i);
        }
    }
}


const double *const
LidDrivenCavity::GetVorticity()
    const
{
    return m_vnew;
}

const double *const
LidDrivenCavity::GetStreamFunction()
    const
{
    return m_s;
}

const double
LidDrivenCavity::GetNx()
    const
{
    return m_Nx;
}

const double
LidDrivenCavity::GetNy()
    const
{
    return m_Ny;
}

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

    m_localArray1 = new double[m_length](); // Local block of first vector
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
}
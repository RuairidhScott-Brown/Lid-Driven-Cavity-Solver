#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>

#define IDX(I, J) ((J)*m_Ny + (I))
// #define IDX(I,J) ((J)*Nx + (I))

#include "../include/LidDrivenCavity.h"
#include "../include/SolverCG.h"

void print_matrix_col(const double *M, int n)
{
    std::cout << "[" << std::endl;
    for (int i{}; i < n; i++)
    {
        for (int j{}; j < n; j++)
        {
            std::cout << M[j * n + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

void print_matrix_row(const double *M, int n)
{
    std::cout << "[" << std::endl;
    for (int i{}; i < n; i++)
    {
        for (int j{}; j < n; j++)
        {
            std::cout << M[i * n + j] << " ";
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

void print_vector(const int *V, int l)
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
            std::cout << "Step: " << setw(8) << t
                    << "  Time: " << setw(8) << t*m_dt
                    << std::endl;
        }
        Advance();
        m_k++;
    }
    MPI_Allgatherv(&m_localArray3[m_rstart], m_rl, MPI_DOUBLE, m_vnew, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    MPI_Allgatherv(&m_localArray1[m_rstart], m_rl, MPI_DOUBLE, m_s, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // if (m_rank == 0) {
    //     std::cout << "After" << std::endl;
    //     print_matrix_row(m_s, 9);
    //     // print_vector(m_localArray1, m_l);
    // }
}

void LidDrivenCavity::ConvertStreamFunctionToVelocityU(double *const u)
{
    for (int i = 1; i < m_Nx - 1; ++i) {
        for (int j = 1; j < m_Ny - 1; ++j) {
            u[IDX(i, j)] = (m_s[IDX(i, j + 1)] - m_s[IDX(i, j)]) / m_dy;
        }
    }
    for (int i = 0; i < m_Nx; ++i) {
        u[IDX(i, m_Ny - 1)] = m_U;
    }
}

void LidDrivenCavity::ConvertStreamFunctionToVelocityV(double *const v)
{
    for (int i = 1; i < m_Nx - 1; ++i) {
        for (int j = 1; j < m_Ny - 1; ++j) {
            v[IDX(i, j)] = -(m_s[IDX(i + 1, j)] - m_s[IDX(i, j)]) / m_dx;
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
    for (int i = 0; i < m_Nx; ++i) {
        for (int j = 0; j < m_Ny; ++j) {
            k = IDX(i, j);
            f << i * m_dx << " " << j * m_dy << " " << m_vnew[k] << " " << m_s[k]
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
    if (m_widths) {
        delete[] m_widths;
        delete[] m_ls;
        delete[] m_rls;
        delete[] m_disp;
        delete[] m_rdisp;
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
        MPI_Sendrecv(&m_localArray2[m_height], m_height, MPI_DOUBLE, m_left, 0, &m_localArray2[m_height * (m_width + 1)], m_height, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArray2[m_height * m_width], m_height, MPI_DOUBLE, m_right, 0, m_localArray2, m_height, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
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
    TimeAdvance(m_localArray1, m_localArray2, m_localArray3);

    // MPI_Allgatherv(&m_localArray3[m_rstart], m_rl, MPI_DOUBLE, m_vnew, m_rls, m_rdisp, MPI_DOUBLE, m_comm);
    // MPI_Allgatherv(&m_localArray1[m_rstart], m_rl, MPI_DOUBLE, m_s, m_rls, m_rdisp, MPI_DOUBLE, m_comm);

    if (m_size > 1) {
        MPI_Sendrecv(&m_localArray3[m_height], m_height, MPI_DOUBLE, m_left, 0, &m_localArray3[m_height*(m_width+1)], m_height, MPI_DOUBLE, m_right, 0, m_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&m_localArray3[m_height*m_width], m_height, MPI_DOUBLE, m_right, 0, m_localArray3, m_height, MPI_DOUBLE, m_left, 0, m_comm, MPI_STATUS_IGNORE);
    }
    // // Solve Poisson problem
    // m_cg->Solve(m_vnew, m_localArray1);

    m_cg->Solve(m_localArray3, m_localArray1);
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

    for (int j = 1; j < m_width + 1; ++j) {
        v[IDX(0, j)] = 2.0 * dx2i * (s[IDX(0, j)] - s[IDX(1, j)]);
        v[IDX(m_height - 1, j)] = 2.0 * dx2i * (s[IDX(m_height - 1, j)] - s[IDX(m_height - 2, j)]);
    }

    if (m_rank == 0) {
        for (int i = 1; i < m_height - 1; ++i) {
            v[IDX(i, 0)] = 2.0 * dy2i * (s[IDX(i, 0)] - s[IDX(i, 1)]);
        }
    }

    if (m_rank == m_size - 1) {
        for (int i = 1; i < m_height - 1; ++i) {
            v[IDX(i, m_width + 1)] = 2.0 * dy2i * (s[IDX(i, m_width + 1)] - s[IDX(i, m_width)]) - 2.0 * dyi * m_U;
        }
    }

    // Compute interior vorticity
    for (int j = 1; j < m_width + 1; ++j) {
        for (int i = 1; i < m_height - 1; ++i) {
            v[IDX(i, j)] = dx2i * (2.0 * s[IDX(i, j)] - s[IDX(i + 1, j)] - s[IDX(i - 1, j)]) + 1.0 / m_dy / m_dy * (2.0 * s[IDX(i, j)] - s[IDX(i, j + 1)] - s[IDX(i, j - 1)]);
        }
    }
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

    for (int j = 1; j < m_width + 1; ++j) {
        for (int i = 1; i < m_height - 1; ++i) {
            v_new[IDX(i, j)] = v[IDX(i, j)] + m_dt * (((s[IDX(i + 1, j)] - s[IDX(i - 1, j)]) * 0.5 * dxi * (v[IDX(i, j + 1)] - v[IDX(i, j - 1)]) * 0.5 * dyi) - ((s[IDX(i, j + 1)] - s[IDX(i, j - 1)]) * 0.5 * dyi * (v[IDX(i + 1, j)] - v[IDX(i - 1, j)]) * 0.5 * dxi) + m_nu * (v[IDX(i + 1, j)] - 2.0 * v[IDX(i, j)] + v[IDX(i - 1, j)]) * dx2i + m_nu * (v[IDX(i, j + 1)] - 2.0 * v[IDX(i, j)] + v[IDX(i, j - 1)]) * dy2i);
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
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_size);
    m_sizeX = std::sqrt(m_size);
    int smallWidthSize{(m_height - 2) / m_size};
    int remainder{(m_height - 2) % m_size};

    m_widths = new int[m_size]();
    m_ls = new int[m_size]();
    m_rls = new int[m_size]();
    m_disp = new int[m_size]();
    m_rdisp = new int[m_size]();

    int offset{};
    int roffset{};

    if (m_rank == 0)
    {
        m_start = 1;
    }
    else if (m_rank == m_size - 1)
    {
        m_end = 1;
    }

    for (int i{}; i < m_size; ++i)
    {
        if (i < remainder)
        {
            m_widths[i] = smallWidthSize + 1;
        }
        else
        {
            m_widths[i] = smallWidthSize;
        }
        m_disp[i] = offset;
        m_rdisp[i] = roffset;
        m_ls[i] = m_widths[i] * m_height + 2 * m_height;

        if (i == 0) {
            m_rls[i] = m_height;
            roffset += m_height;
        }
        if (i == m_size - 1) {
            m_rls[i] += m_height;
        }  
        m_rls[i] += m_widths[i] * m_height;
        roffset += (m_widths[i]) * m_height;

        offset += (m_widths[i]) * m_height;
    }

    m_width = m_widths[m_rank];
    m_l = m_width * m_height + 2 * m_height;
    m_rl = m_rls[m_rank];

    m_localArray1 = new double[m_l](); // Local block of first vector
    m_localArray2 = new double[m_l]();
    m_localArray3 = new double[m_l]();
    m_localArray4 = new double[m_l]();
    // m_localArrayX = new double[m_l]();
    // m_localArrayR = new double[m_l]();
    // m_localArrayZ = new double[m_l]();
    // m_localArrayB = new double[m_l]();

    m_left = m_rank - 1;
    m_right = m_rank + 1;
    if (m_start)
    {
        m_left = MPI_PROC_NULL;
    }
    else if (m_end)
    {
        m_right = MPI_PROC_NULL;
    }

    if (m_start)
    {
        m_rstart = 0;
    }
    else
    {
        m_rstart = m_height;
    }
    // MPI_Scatterv(m_s, m_ls, m_disp, MPI_DOUBLE, m_localArray1, m_l, MPI_DOUBLE, 0, m_comm);
    // MPI_Scatterv(m_v, m_ls, m_disp, MPI_DOUBLE, m_localArray2, m_l, MPI_DOUBLE, 0, m_comm);
    // MPI_Scatterv(m_vnew, m_ls, m_disp, MPI_DOUBLE, m_localArray3, m_l, MPI_DOUBLE, 0, m_comm);
}
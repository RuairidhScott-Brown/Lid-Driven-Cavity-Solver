#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>

#define IDX(I,J) ((J)*m_Nx + (I))

#include "../include/LidDrivenCavity.h"
#include "../include/SolverCG.h"

LidDrivenCavity::LidDrivenCavity()
{
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
    m_nu = 1.0/re;
}

void LidDrivenCavity::SetRank(int rankRow, int rankCol)
{
    m_rankRow = rankRow;
    m_rankCol = rankCol;
}

void LidDrivenCavity::SetSize(int size)
{
    m_size = size;
    m_sizeX = sqrt(m_size);
}

void LidDrivenCavity::SetCommunicator(MPI_Comm grid) 
{
    m_grid = grid;
}

void LidDrivenCavity::Initialise()
{
    CleanUp();

    
    m_v   = new double[m_Npts]();
    m_vnew = new double[m_Npts]();
    m_vtemp = new double[m_Npts]();
    m_s   = new double[m_Npts]();
    m_tmp = new double[m_Npts]();

    m_lNx = m_Nx / m_sizeX;
    m_lNy = m_Ny / m_sizeX;

    m_lrNx = m_Nx % m_sizeX;
    m_lrNy = m_Ny % m_sizeX;


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


    switch (m_lrNx) {
        case 0:
            m_startNx = m_rankCol*m_lNx;
            m_endNx = m_startNx + m_lNx-1;
            break;
        case 1:
            m_startNx = m_rankCol*m_lNx;
            m_endNx = m_startNx + m_lNx-1 + lastX;
            break;
        case 2:
            m_startNx = m_rankCol*m_lNx + lastX;
            m_endNx = m_startNx + m_lNx-1 + secondLastX;
            break;
        case 3:
            m_startNx = m_rankCol*m_lNx + secondLastX + lastX;
            m_endNx = m_startNx + m_lNx-1 + thirdLastX;
            break;
    }

    switch (m_lrNy) {
        case 0:
            m_startNy = m_rankRow*m_lNy;
            m_endNy = m_startNy + m_lNy-1;
            break;
        case 1:
            m_startNy = m_rankRow*m_lNy;
            m_endNy = m_startNy + m_lNy-1 + lastY;
            break;
        case 2:
            m_startNy = m_rankRow*m_lNy + lastY;
            m_endNy = m_startNy + m_lNy-1 + secondLastY;
            break;
        case 3:
            m_startNy = m_rankCol*m_lNy + secondLastY + lastY;
            m_endNy = m_startNy + m_lNy-1 + thirdLastY;
            break;
    }

    m_endNx++;
    m_endNy++;

    if (m_rankCol == 0) m_startNx++;
    if (m_rankCol == (m_sizeX-1)) m_endNx--;

    if (m_rankRow == 0) m_startNy++;
    if (m_rankRow == (m_sizeX-1)) m_endNy--;
    // std::cout << std::endl;
    // std::cout <<"(" << m_rankRow << "," << m_rankCol << ") " << "Start: x " << m_startNx << " End: x " << m_endNx << std::endl;
    // std::cout <<"(" << m_rankRow << "," << m_rankCol << ") " << "Start: y " << m_startNy << " End: y " << m_endNy << std::endl;

    m_cg  = new SolverCG(m_Nx, m_Ny, m_dx, m_dy);
    m_cg->SetCommunicator();
    // m_cg->SetRank(m_rankRow, m_rankCol);
    // m_cg->SetSubGridDimensions(m_startNx, m_endNx, m_startNy, m_endNy);
}

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(m_T/m_dt);

    for (int t = 0; t < NSteps; ++t)
    {
        if(m_rankRow == 0 && m_rankCol == 0) {
            // std::cout << "Step: " << setw(8) << t
            //         << "  Time: " << setw(8) << t*m_dt
            //         << std::endl;
        }
        Advance();
    }
}

void LidDrivenCavity::ConvertStreamFunctionToVelocityU(double* const u)
{
    for (int i = 1; i < m_Nx - 1; ++i) {
        for (int j = 1; j < m_Ny - 1; ++j) {
            u[IDX(i,j)] =  (m_s[IDX(i,j+1)] - m_s[IDX(i,j)]) / m_dy;
        }
    }
    for (int i = 0; i < m_Nx; ++i) {
        u[IDX(i,m_Ny-1)] = m_U;
    }
}

void LidDrivenCavity::ConvertStreamFunctionToVelocityV(double* const v)
{
    for (int i = 1; i < m_Nx - 1; ++i) {
        for (int j = 1; j < m_Ny - 1; ++j) {
            v[IDX(i,j)] = -(m_s[IDX(i+1,j)] - m_s[IDX(i,j)]) / m_dx;
        }
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{

    if (m_rankRow != 0 || m_rankCol != 0) return;

    double* u0 = new double[m_Nx*m_Ny]();
    double* u1 = new double[m_Nx*m_Ny]();

    ConvertStreamFunctionToVelocityU(u0);
    ConvertStreamFunctionToVelocityV(u1);

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            k = IDX(i, j);
            f << i * m_dx << " " << j * m_dy << " " << m_vnew[k] <<  " " << m_s[k] 
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
    cout << "Steps:     " << ceil(m_T/m_dt) << endl;
    cout << "Reynolds number: " << m_Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << "Size:      " << m_size << endl; 
    cout << endl;
    if (m_nu * m_dt / m_dx / m_dy > 0.25) {
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
}


void LidDrivenCavity::UpdateDxDy()
{
    m_dx = m_Lx / (m_Nx-1);
    m_dy = m_Ly / (m_Ny-1);
    m_Npts = m_Nx * m_Ny;
}


void LidDrivenCavity::Advance()
{
    double dxi  = 1.0/m_dx;
    double dyi  = 1.0/m_dy;
    double dx2i = 1.0/m_dx/m_dx;
    double dy2i = 1.0/m_dy/m_dy;


    std::memset(m_v, 0, m_Nx*m_Ny*sizeof(double));
    std::memset(m_vnew, 0, m_Nx*m_Ny*sizeof(double));
    std::memset(m_vtemp, 0, m_Nx*m_Ny*sizeof(double));

    if (m_rankRow == 0) {
        for (int i = m_startNx; i < m_endNx; ++i) {
        // Bottom
            m_vtemp[IDX(i,0)]    = 2.0 * dy2i * (m_s[IDX(i,0)]    - m_s[IDX(i,1)]);
        }
    }
    if (m_rankRow == (m_sizeX-1)) {
        for (int i = m_startNx; i < m_endNx; ++i) {
            m_vtemp[IDX(i,m_Ny-1)] = 2.0 * dy2i * (m_s[IDX(i,m_Ny-1)] - m_s[IDX(i,m_Ny-2)])
                        - 2.0 * dyi*m_U;
        }
    }

    if (m_rankCol == 0) {
        for (int j = m_startNy; j < m_endNy; ++j) {
            // left
            m_vtemp[IDX(0,j)]    = 2.0 * dx2i * (m_s[IDX(0,j)]    - m_s[IDX(1,j)]);
        }
    }

    if (m_rankCol == (m_sizeX-1)) {
        for (int j = m_startNy; j < m_endNy; ++j) {
            // right
            m_vtemp[IDX(m_Nx-1,j)] = 2.0 * dx2i * (m_s[IDX(m_Nx-1,j)] - m_s[IDX(m_Nx-2,j)]);
        }
    }

    for (int i = m_startNx; i < m_endNx; ++i) {
        for (int j = m_startNy; j < m_endNy; ++j) {
            m_vtemp[IDX(i,j)] = dx2i*(
                    2.0 * m_s[IDX(i,j)] - m_s[IDX(i+1,j)] - m_s[IDX(i-1,j)])
                        + 1.0/m_dy/m_dy*(
                    2.0 * m_s[IDX(i,j)] - m_s[IDX(i,j+1)] - m_s[IDX(i,j-1)]);
        }
    }

    MPI_Allreduce(m_vtemp, m_v, m_Nx*m_Ny, MPI_DOUBLE, MPI_SUM, m_grid);
    std::memset(m_vtemp, 0, m_Nx*m_Ny*sizeof(double));

    for (int i = m_startNx; i < m_endNx; ++i) {
        for (int j = m_startNy; j < m_endNy; ++j) {
            m_vtemp[IDX(i,j)] = m_v[IDX(i,j)] + m_dt*(
                ( (m_s[IDX(i+1,j)] - m_s[IDX(i-1,j)]) * 0.5 * dxi
                 *(m_v[IDX(i,j+1)] - m_v[IDX(i,j-1)]) * 0.5 * dyi)
              - ( (m_s[IDX(i,j+1)] - m_s[IDX(i,j-1)]) * 0.5 * dyi
                 *(m_v[IDX(i+1,j)] - m_v[IDX(i-1,j)]) * 0.5 * dxi)
              + m_nu * (m_v[IDX(i+1,j)] - 2.0 * m_v[IDX(i,j)] + m_v[IDX(i-1,j)])*dx2i
              + m_nu * (m_v[IDX(i,j+1)] - 2.0 * m_v[IDX(i,j)] + m_v[IDX(i,j-1)])*dy2i);
        }
    }

    MPI_Allreduce(m_vtemp, m_vnew, m_Nx*m_Ny, MPI_DOUBLE, MPI_SUM, m_grid);

    // Solve Poisson problem
    m_cg->Solve(m_vnew, m_s);
}


const double* const 
LidDrivenCavity::GetVorticity() 
const
{
    return m_vnew;
}

const double* const 
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
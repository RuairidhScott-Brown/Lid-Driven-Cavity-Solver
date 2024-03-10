#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>

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

void LidDrivenCavity::Initialise()
{
    CleanUp();

    m_v   = new double[m_Npts]();
    m_s   = new double[m_Npts]();
    m_tmp = new double[m_Npts]();
    m_cg  = new SolverCG(m_Nx, m_Ny, m_dx, m_dy);
}

void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(m_T/m_dt);
    for (int t = 0; t < NSteps; ++t)
    {
        std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*m_dt
                  << std::endl;
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
            f << i * m_dx << " " << j * m_dy << " " << m_v[k] <<  " " << m_s[k] 
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

    // Boundary node vorticity
    for (int i = 1; i < m_Nx-1; ++i) {
        // Bottom
        m_v[IDX(i,0)]    = 2.0 * dy2i * (m_s[IDX(i,0)]    - m_s[IDX(i,1)]);
        // Top
        m_v[IDX(i,m_Ny-1)] = 2.0 * dy2i * (m_s[IDX(i,m_Ny-1)] - m_s[IDX(i,m_Ny-2)])
                       - 2.0 * dyi*m_U;
    }
    for (int j = 1; j < m_Ny-1; ++j) {
        // left
        m_v[IDX(0,j)]    = 2.0 * dx2i * (m_s[IDX(0,j)]    - m_s[IDX(1,j)]);
        // right
        m_v[IDX(m_Nx-1,j)] = 2.0 * dx2i * (m_s[IDX(m_Nx-1,j)] - m_s[IDX(m_Nx-2,j)]);
    }

    // Compute interior vorticity
    for (int i = 1; i < m_Nx - 1; ++i) {
        for (int j = 1; j < m_Ny - 1; ++j) {
            m_v[IDX(i,j)] = dx2i*(
                    2.0 * m_s[IDX(i,j)] - m_s[IDX(i+1,j)] - m_s[IDX(i-1,j)])
                        + 1.0/m_dy/m_dy*(
                    2.0 * m_s[IDX(i,j)] - m_s[IDX(i,j+1)] - m_s[IDX(i,j-1)]);
        }
    }

    // Time advance vorticity
    for (int i = 1; i < m_Nx - 1; ++i) {
        for (int j = 1; j < m_Ny - 1; ++j) {
            m_v[IDX(i,j)] = m_v[IDX(i,j)] + m_dt*(
                ( (m_s[IDX(i+1,j)] - m_s[IDX(i-1,j)]) * 0.5 * dxi
                 *(m_v[IDX(i,j+1)] - m_v[IDX(i,j-1)]) * 0.5 * dyi)
              - ( (m_s[IDX(i,j+1)] - m_s[IDX(i,j-1)]) * 0.5 * dyi
                 *(m_v[IDX(i+1,j)] - m_v[IDX(i-1,j)]) * 0.5 * dxi)
              + m_nu * (m_v[IDX(i+1,j)] - 2.0 * m_v[IDX(i,j)] + m_v[IDX(i-1,j)])*dx2i
              + m_nu * (m_v[IDX(i,j+1)] - 2.0 * m_v[IDX(i,j)] + m_v[IDX(i,j-1)])*dy2i);
        }
    }

    // Sinusoidal test case with analytical solution, which can be used to test
    // the Poisson solver
    /*
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < m_Nx; ++i) {
        for (int j = 0; j < m_Ny; ++j) {
            m_v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * m_dx)
                                       * sin(M_PI * l * j * m_dy);
        }
    }
    */

    // Solve Poisson problem
    m_cg->Solve(m_v, m_s);
}


const double* const 
LidDrivenCavity::GetVorticity() 
const
{
    return m_v;
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
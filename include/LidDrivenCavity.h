#pragma once

#include <string>
#include <mpi.h>
using namespace std;

class SolverCG;

class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);
    void SetRank(int rankRow, int rankCol);
    void SetSize(int size);
    void SetCommunicator(MPI_Comm grid);

    void Initialise();
    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();

    void ConvertStreamFunctionToVelocityU(double* const u);
    void ConvertStreamFunctionToVelocityV(double* const v);

    const double* const GetVorticity() const;
    const double* const GetStreamFunction() const;

    const double GetNy() const;
    const double GetNx() const;


private:
    double* m_v     {};
    double* m_vnew  {};
    double* m_vtemp {};
    double* m_s     {};
    double* m_tmp   {};

    double m_dt     {0.01};
    double m_T      {1.0};
    double m_dx     {};
    double m_dy     {};
    int    m_Nx     {9};
    int    m_Ny     {9};
    int    m_lNx    {};
    int    m_lNy    {};
    int    m_lrNx    {};
    int    m_lrNy    {};
    int    m_startNx {};
    int    m_startNy {};
    int    m_endNx  {};
    int    m_endNy  {};
    int    m_Npts   {81};
    double m_Lx     {1.0};
    double m_Ly     {1.0};
    double m_Re     {10};
    double m_U      {1.0};
    double m_nu     {0.1};

    int m_rankRow    {};
    int m_rankCol    {};
    int m_size       {};
    int m_sizeX      {};
    int k {};

    SolverCG* m_cg {};
    MPI_Comm m_grid {};

    void CleanUp();
    void UpdateDxDy();
    void Advance();
};


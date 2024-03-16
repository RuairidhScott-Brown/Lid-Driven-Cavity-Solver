#pragma once

#include <string>
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
    double* m_s     {};
    double* m_tmp   {};

    double m_dt     {0.01};
    double m_T      {1.0};
    double m_dx     {};
    double m_dy     {};
    int    m_Nx     {9};
    int    m_Ny     {9};
    int    m_Npts   {81};
    double m_Lx     {1.0};
    double m_Ly     {1.0};
    double m_Re     {10};
    double m_U      {1.0};
    double m_nu     {0.1};

    SolverCG* m_cg {};

    void CleanUp();
    void UpdateDxDy();
    void Advance();
};


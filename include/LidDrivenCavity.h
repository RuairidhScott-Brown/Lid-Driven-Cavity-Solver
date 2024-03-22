#pragma once

#include <string>
#include <mpi.h>
using namespace std;

enum class LidDrivenCavityConfigError {
    SUCCESS,
    FAILED
};


class SolverCG;

/**
 * @brief Object responsible for the
 * solving of the Lid Driven Cavity problem.
 * 
 */
class LidDrivenCavity
{
public:
    LidDrivenCavity(MPI_Comm comm = MPI_COMM_WORLD);
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);

    void Initialise();
    void Integrate();
    void WriteSolution(std::string file);
    LidDrivenCavityConfigError PrintConfiguration();
    LidDrivenCavityConfigError CheckConfiguration();


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
 
    int    m_Npts   {81};
    double m_Lx     {1.0};
    double m_Ly     {1.0};
    double m_Re     {10};
    double m_U      {1.0};
    double m_nu     {0.1};
    int m_k {};

    double m_dxi {};
    double m_dyi {};
    double m_dx2i {};
    double m_dy2i {};
    double m_2dx2i {};
    double m_2dy2i {};

    int m_rankRow    {};
    int m_rankCol    {};
    int m_rank       {};
    int m_size       {};
    int m_sizeX      {};

    int m_localHeightPlusOne {};
    int m_widthMinusOne {};

    // Dimensions of the local matrix
    int* m_localHeights {};
    int* m_lengths {};
    int* m_returnLengths {};
    int* m_displacements {};
    int* m_returnDisplacements {};
    int m_height {};
    int m_width {};
    int m_start {};
    int m_end   {};
    int m_localHeight {};
    int m_length {};
    int m_returnLength {};

    // Temporary data arrays.
    double* m_localArray1 {};
    double* m_localArray2 {};
    double* m_localArray3 {};
    double* m_localArray4 {};

    int m_left {};
    int m_right {};
    int m_returnStart {};

    SolverCG* m_cg {};
    MPI_Comm m_comm {};

    void CleanUp();
    void UpdateDxDy();
    void Advance();
    void SetSize();
    void TimeAdvance(double* s, double* v, double* v_new);
    void Vorticity(double* s, double* v);
};


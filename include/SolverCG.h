#pragma once

#include <mpi.h>

enum class SolverCGErrorCode {
    SUCCESS,
    CONVERGE_FAILED
};

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();

    void SetCommunicator(MPI_Comm grid);
    void SetRank(int rankRow, int rankCol);
    void SetSubGridDimensions(int startNx, int endNx, int startNy, int endNy);
    void CalculateSubGridDimensions();



    SolverCGErrorCode Solve(double* b, double* x);

private:
    double m_dx     {};
    double m_dy     {};
    int m_Nx        {};
    int m_Ny        {};
    double* m_r     {};
    double* m_p     {};
    double* m_z     {};
    double* m_t     {};
    double* m_t2     {};
    int*    m_arrays {};
    double* m_localArray1 {}; // Local block of first vector
    double* m_localArray2 {};
    int* m_disp {};

    MPI_Comm m_grid {};

    int m_rankRow   {};
    int m_rankCol   {};
    int m_globalRank {};
    int m_size      {};
    int m_sizeX     {};
    int m_startNx   {};
    int m_startNy   {};
    int m_endNx     {};
    int m_endNy     {};
    int m_localSize {};
    int m_k {};
    bool m_useMPI   {false};

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);
    double MPI_cblas_ddot(const int m, const double* const x, const double* const y);
    double MPI_cblas_dnrm2(const int m, const double* const x);
    void MPI_cblas_daxpy(const int m, const double alpha, double* const x,  double* const y);


    void SetSize(int size);

};


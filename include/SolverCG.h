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
    void SetSize(int size);
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

    MPI_Comm m_grid {};

    int m_rankRow   {};
    int m_rankCol   {};
    int m_size      {};
    int m_sizeX     {};
    int m_startNx   {};
    int m_startNy   {};
    int m_endNx     {};
    int m_endNy     {};
    int m_k {};
    bool m_useMPI   {false};

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};


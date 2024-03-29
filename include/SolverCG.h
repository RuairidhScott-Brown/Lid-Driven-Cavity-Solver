#pragma once

#include <mpi.h>

enum class SolverCGErrorCode {
    SUCCESS,
    CONVERGE_FAILED
};

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy, MPI_Comm comm = MPI_COMM_WORLD);
    ~SolverCG();

    // void SetCommunicator();

    SolverCGErrorCode Solve(double* b, double* x);

private:
    double m_dx     {};
    double m_dy     {};
    int m_Nx        {};
    int m_Ny        {};


    double* m_localArrayP {}; // Local block of first vector
    double* m_localArrayT {};
    double* m_localArrayX {};
    double* m_localArrayR {};
    double* m_localArrayZ {};
    double* m_localArrayB {};
    double* m_localPre {};
    double* m_localBC {};
    int* m_disp {};
    int m_solver_rank {-1}; 
    int m_solver_size {-1};

    MPI_Comm m_solver_comm {};

    int m_k {};
    int m_height {};
    int m_width {};
    int* m_widths {};
    int* m_ls {};
    int* m_rls {};
    int* m_rdisp {};
    int m_end {};
    int m_start {};
    int m_l {};
    int m_rl {};
    int m_left {-1};
    int m_right {-1};
    int m_rstart {};


    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);
    double MPI_cblas_ddot(const int m, const double* const x, const double* const y);
    double MPI_cblas_dnrm2(const int m, const double* const x);
    void MPI_cblas_daxpy(const int m, const double alpha, double* const x,  double* const y);
    void MPI_cblas_dcopy(const int m, double* const x, double* const y);
    void PopulatePreconditionBandedMatrix();
    void PopulateImposeBCBandedMatrix();
    void MPI_Precondition(double* in, double* out);
    void CreateMatrices();
    void MPI_ImposeBC(double* out);
    void MPI_ApplyOperator(double* in, double* out);
    void Laplace(double* in, double* out);
    SolverCGErrorCode SolveWithMultipleRank(double* b, double* x);
    SolverCGErrorCode SolveWithSingleRank(double* b, double* x);
    void SetSize();

};


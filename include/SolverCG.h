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

    void SetCommunicator();
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
    double* m_t2    {};
    double* m_pre   {};
    double* m_bc    {};
    double* m_localArray1 {}; // Local block of first vector
    double* m_localArray2 {};
    double* m_localArrayP {}; // Local block of first vector
    double* m_localArrayT {};
    double* m_localArrayX {};
    double* m_localArrayR {};
    double* m_localArrayZ {};
    double* m_localArrayB {};
    double* m_localPre {};
    double* m_localBC {};
    double* m_local {};
    int* m_disp {};
    int m_solver_rank {-1}; 
    int m_solver_size {-1};

    MPI_Comm m_grid {};
    MPI_Comm m_solver_comm {};

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
    int* m_arrays {};
    int* m_disp2 {};
    int m_k {};
    bool m_useMPI   {false};
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




    






    void SetSize(int size);

};


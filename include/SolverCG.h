#pragma once

#include <mpi.h>

enum class SolverCGErrorCode {
    SUCCESS,
    CONVERGE_FAILED
};

/**
 * @brief Object responsible for the solving of the
 * Poissons problem.
 * 
 */
class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy, MPI_Comm comm = MPI_COMM_WORLD);
    ~SolverCG();
    void print_matrix_row_major(double* M, int r, int c);


    void UseMPI(bool yes);

    SolverCGErrorCode Solve(double* b, double* x);

private:
    double m_dx     {};
    double m_dy     {};
    int m_Nx        {};
    int m_Ny        {};
    int m_localHeight {};
    int m_length {};
    int m_returnLength {};

    int m_localHeightPlusOne {};
    int m_widthMinusOne {};

    double m_dx2i {};
    double m_dy2i {};
    double m_factor {};
    int m_k {};
    double m_alpha {};
    double m_beta {};
    double m_eps {};
    double m_tol {0.001*0.001};
    double m_localDotProductAplha {};
    double m_localDotProductBeta {};
    double m_globalDotProductEps {};
    double m_localDotProductEps {};
    double m_localDotProductTemp {};
    double m_globalDotProductTemp {};



    double* m_localArrayP {}; // Local block of first vector
    double* m_localArrayT {};
    double* m_localArrayX {};
    double* m_localArrayR {};
    double* m_localArrayZ {};
    double* m_localArrayB {};
    double* m_localPre {};
    double* m_localBC {};
    int* m_displacements {};
    int m_solver_rank {-1}; 
    int m_solver_size {-1};

    MPI_Comm m_solver_comm {};

    int m_height {};
    int m_width {};
    int* m_localHeights {};
    int* m_lengths {};
    int* m_returnLengths {};
    int* m_returnDisplacements {};
    int m_end {};
    int m_start {};
    int m_left {-1};
    int m_right {-1};
    int m_returnStart {};


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


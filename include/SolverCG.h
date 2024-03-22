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
    // Constants needed for the equations.
    double m_dx     {};
    double m_dy     {};
    int m_Nx        {};
    int m_Ny        {};
    double m_dx2i {};
    double m_dy2i {};
    double m_factor {};

    // Dimensions of the local matrix.
    int m_localHeight {};
    int m_length {};
    int m_returnLength {};

    int m_localHeightPlusOne {};
    int m_widthMinusOne {};

    int m_k {};
    double m_alpha {};
    double m_beta {};
    double m_eps {};
    double m_tol {0.001*0.001};

    // Required for MPI reduce.
    double m_localDotProductAplha {};
    double m_localDotProductBeta {};
    double m_globalDotProductEps {};
    double m_localDotProductEps {};
    double m_localDotProductTemp {};
    double m_globalDotProductTemp {};


    // Temporary arrays.
    double* m_localArrayP {}; 
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

    // Communicators.
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


    void CreateMatrices();
    void Laplace(double* in, double* out);
    void SetSize();

    SolverCGErrorCode SolveWithMultipleRank(double* b, double* x);
    SolverCGErrorCode SolveWithSingleRank(double* b, double* x);
};


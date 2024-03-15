#pragma once

enum class SolverCGErrorCode {
    SUCCESS,
    CONVERGE_FAILED
};

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();

    SolverCGErrorCode Solve(double* b, double* x);

private:
    double m_dx;
    double m_dy;
    int m_Nx;
    int m_Ny;
    double* m_r;
    double* m_p;
    double* m_z;
    double* m_t;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};


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
    double dx;
    double dy;
    int Nx;
    int Ny;
    double* r;
    double* p;
    double* z;
    double* t;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};


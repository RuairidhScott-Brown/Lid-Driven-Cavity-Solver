#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE Main
#include <boost/test/included/unit_test.hpp>
#include <sstream>
#include <fstream>
#include <vector>
#include <experimental/filesystem>
#include <cmath>

#include "../include/SolverCG.h"
#include "../include/LidDrivenCavity.h"


struct DefaultLidDrivenCavity {
    int Nx {9};
    int Ny {9};
    double Lx {1.0};
    double Ly {1.0};
    double dx {1/8};
    double dy {1/8};
    double dt {0.01};
    double T {1};
    double Re {10};
};

struct DefaultLidDrivenCavityData {
    double x[81] {}; 
    double y[81] {}; 
    double vorticity[81] {};
    double streamFunction[81] {};
    double u[81] {};
    double v[81] {};
};

double 
RoundSigFig(const double x, const int n)
{ 
    char buff[32];
    sprintf(buff, "%.*g", n, x);
    return atof(buff);
}

bool 
AllValuesAreZero(const double* arr, const size_t size) 
{
    for(size_t i {}; i < size; ++i) {
        if (arr[i] != 0.0) {
            return false;
        }
    }
    return true;
}

bool 
ArraysAreEqual(const double* arr1, const double* arr2, const size_t size, const int sig) 
{
    for(size_t i {}; i < size; ++i) {
        if(std::abs(RoundSigFig(arr1[i], sig) - RoundSigFig(arr2[i], sig)) > 1.0e-6) {
            std::cout << "Error :" << std::setprecision(10) << arr1[i] << " " <<  arr2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void 
ReadDefaultDataFromFile(const std::string& filename, DefaultLidDrivenCavityData* const data) 
{
    std::ifstream file(filename);
    std::string line;

    DefaultLidDrivenCavity dldc {};

    double x {}; 
    double y {}; 
    double vorticity {};
    double streamFunction {};
    double u {};
    double v {};


    for(int i {}; i < dldc.Nx; ++i) {
        for(int j {}; j < dldc.Ny; ++j) {
            if (!std::getline(file, line)) break;
            std::istringstream iss(line);
            if(iss >> x >> y >> vorticity >> streamFunction >> u >> v) {
                data->x[j*dldc.Ny + i] = x;
                data->y[j*dldc.Ny + i] = y;
                data->vorticity[j*dldc.Ny + i] = vorticity;
                data->streamFunction[j*dldc.Ny + i] = streamFunction;
                data->u[j*dldc.Ny + i] = u;
                data->v[j*dldc.Ny + i] = v;
            } else {
                j--;
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TestSolverCGAllZeros)
{
    DefaultLidDrivenCavity dldc {};

    int n {dldc.Nx*dldc.Ny};
    double b[n] {};
    double x[n] {};

    SolverCG solver {dldc.Nx, dldc.Ny, dldc.dx, dldc.dy};

    SolverCGErrorCode result {solver.Solve(b, x)};

    BOOST_TEST(AllValuesAreZero(x, n) == true);
    BOOST_CHECK(result == SolverCGErrorCode::SUCCESS);
}

BOOST_AUTO_TEST_CASE(TestSolverCGAllOnes)
{
    DefaultLidDrivenCavity dldc {};

    int n {dldc.Nx*dldc.Ny};
    double b[n] {1};
    double x[n] {};

    SolverCG solver {dldc.Nx, dldc.Ny, dldc.dx, dldc.dy};

    SolverCGErrorCode result {solver.Solve(b, x)};

    BOOST_CHECK(result == SolverCGErrorCode::CONVERGE_FAILED);
}


BOOST_AUTO_TEST_CASE(TestLidDrivenCavity)
{
    DefaultLidDrivenCavity dldc {};

    LidDrivenCavity solver {LidDrivenCavity()};

    int n {dldc.Nx*dldc.Ny};

    solver.SetDomainSize(dldc.Lx, dldc.Ly);
    solver.SetGridSize(dldc.Nx, dldc.Ny);
    solver.SetTimeStep(dldc.dt);
    solver.SetFinalTime(dldc.T);
    solver.SetReynoldsNumber(dldc.Re);

    solver.Initialise();
    solver.Integrate();

    double u[n] {};
    double v[n] {};

    const double* const vorticity {solver.GetVorticity()};
    const double* const streamFunction {solver.GetStreamFunction()};

    solver.ConvertStreamFunctionToVelocityU(u);
    solver.ConvertStreamFunctionToVelocityV(v);

    std::string srcPath = std::experimental::filesystem::canonical(__FILE__).string();
    std::string srcDir = srcPath.substr(0, srcPath.find_last_of("/\\"));
    std::string filePath = srcDir + "/data/default_solution.txt";

    DefaultLidDrivenCavityData data {};
    ReadDefaultDataFromFile(filePath, &data);

    int sigFig {6};
    BOOST_TEST(ArraysAreEqual(u, data.u, n, sigFig) == true);
    BOOST_TEST(ArraysAreEqual(v, data.v, n, sigFig) == true);
    BOOST_TEST(ArraysAreEqual(vorticity, data.vorticity, n, sigFig) == true);
    BOOST_TEST(ArraysAreEqual(streamFunction, data.streamFunction, n, sigFig) == true);
}

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE Main
#include <boost/test/included/unit_test.hpp>
#include <sstream>
#include <fstream>
#include <vector>
#include <experimental/filesystem>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <mpi.h>

#define IDX(I,J) ((J)*Nx + (I))

#include "../include/SolverCG.h"
#include "../include/LidDrivenCavity.h"

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

struct MPIFixture {
    public:
        explicit MPIFixture() {
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout << "Initialising MPI" << endl;
            MPI_Init(&argc, &argv);
        }

        ~MPIFixture() {
            cout << "Finalising MPI" << endl;
            MPI_Finalize();
        }

        int argc;
        char **argv;
};
BOOST_GLOBAL_FIXTURE(MPIFixture);

struct DefaultLidDrivenCavity {
    int Nx {9};
    int Ny {9};
    double Lx {1.0};
    double Ly {1.0};
    double dx {1.0/8.0};
    double dy {1.0/8.0};
    double dt {0.01};
    double T {1};
    double Re {10};
};

struct DefaultLidDrivenCavityData {
    double* x;
    double* y;
    double* vorticity;
    double* streamFunction;
    double* u;
    double* v;
    int size;

    DefaultLidDrivenCavityData(int arraySize = 81) : size(arraySize) {
        x = new double[size];
        y = new double[size];
        vorticity = new double[size];
        streamFunction = new double[size];
        u = new double[size];
        v = new double[size];
    }

    ~DefaultLidDrivenCavityData() {
        delete[] x;
        delete[] y;
        delete[] vorticity;
        delete[] streamFunction;
        delete[] u;
        delete[] v;
    }
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
ArraysAreEqual(const double* arr1, const double* arr2, const size_t size) 
{
    for(size_t i {}; i < size; ++i) {
        // if(std::abs(RoundSigFig(arr1[i] + offset, sig) - RoundSigFig(arr2[i] + offset, sig)) > 1.0e-6) {
        //     std::cout << "Error :" << std::setprecision(10) << arr1[i] << " " <<  arr2[i] << std::endl;
        //     return false;
        // }
        BOOST_TEST(arr1[i] == arr2[i], tt::tolerance(0.0001));
    }
    return true;
}

void 
ReadDefaultDataFromFile(const std::string& filename, DefaultLidDrivenCavityData* const data, int Nx = 9, int Ny = 9) 
{
    std::ifstream file(filename);
    std::string line;

    double x {}; 
    double y {}; 
    double vorticity {};
    double streamFunction {};
    double u {};
    double v {};


    for(int i {}; i < Nx; ++i) {
        for(int j {}; j < Ny; ++j) {
            if (!std::getline(file, line)) break;
            std::istringstream iss(line);
            if(iss >> x >> y >> vorticity >> streamFunction >> u >> v) {
                data->x[j*Ny + i] = x;
                data->y[j*Ny + i] = y;
                data->vorticity[j*Ny + i] = vorticity;
                data->streamFunction[j*Ny + i] = streamFunction;
                data->u[j*Ny + i] = u;
                data->v[j*Ny + i] = v;
            } else {
                j--;
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TestSolverCG){

    DefaultLidDrivenCavity dldc {};

    SolverCG solver(dldc.Nx, dldc.Ny, dldc.dx, dldc.dy);
    solver.UseMPI(false);

    int n {dldc.Nx*dldc.Ny};
    int Nx {dldc.Nx};
    double b[n] {};
    double x[n] {};
    double expected_x[n] {};

    const int k = 3;
    const int l = 3;

    for (int i {}; i < dldc.Nx; ++i) {
        for (int j {}; j < dldc.Ny; ++j) {
            expected_x[IDX(i, j)] = -M_PI*M_PI*(k*k + l*l)*sin(M_PI*k*i*dldc.dx)*sin(M_PI*l*j*dldc.dy);
        }
    }

    for (int i {1}; i < Nx-1; ++i) {
        for (int j {1}; j < dldc.Ny-1; ++j) {
            b[IDX(i, j)] = -1.0*((expected_x[IDX(i-1, j)] - 2.0*expected_x[IDX(i, j)] + expected_x[IDX(i+1, j)]) / (dldc.dx*dldc.dx)
                        + (expected_x[IDX(i, j-1)] - 2.0*expected_x[IDX(i, j)] + expected_x[IDX(i, j+1)]) / (dldc.dy*dldc.dy));
        }
    }

    solver.Solve(b, x);

    BOOST_TEST(ArraysAreEqual(x, expected_x, n) == true);
}

BOOST_AUTO_TEST_CASE(TestSolverCGConverFailed)
{
    DefaultLidDrivenCavity dldc {};

    int n {dldc.Nx*dldc.Ny};
    double b[n] {1};
    double x[n] {};

    SolverCG solver(dldc.Nx, dldc.Ny, dldc.dx, dldc.dy);

    SolverCGErrorCode result {solver.Solve(b, x)};
    BOOST_CHECK(result == SolverCGErrorCode::CONVERGE_FAILED);
}

BOOST_AUTO_TEST_CASE(TestSolverCGAllZeros)
{
    DefaultLidDrivenCavity dldc {};

    int n {dldc.Nx*dldc.Ny};
    double b[n] {};
    double x[n] {};

    SolverCG solver(dldc.Nx, dldc.Ny, dldc.dx, dldc.dy);

    SolverCGErrorCode result {solver.Solve(b, x)};

    BOOST_TEST(AllValuesAreZero(x, n) == true);
    BOOST_CHECK(result == SolverCGErrorCode::SUCCESS);
}

BOOST_AUTO_TEST_CASE(TestLidDrivenCavity)
{
    DefaultLidDrivenCavity dldc {};

    int worldSize {};
    int worldRank {};

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm grid;
    const int dims {2};
    const int nAxisRanks {sqrt(worldSize)};
    int sizes[dims] = {nAxisRanks, nAxisRanks};
    int periods[dims] = {0, 1};
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, dims, sizes, periods, reorder, &grid);

    int coords[dims];
    int gridRank {};

    MPI_Comm_rank(grid, &gridRank);
    MPI_Cart_coords(grid, gridRank, dims, coords);

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

    BOOST_TEST(ArraysAreEqual(u, data.u, n) == true);
    BOOST_TEST(ArraysAreEqual(v, data.v, n) == true);
    BOOST_TEST(ArraysAreEqual(vorticity, data.vorticity, n) == true);
    BOOST_TEST(ArraysAreEqual(streamFunction, data.streamFunction, n) == true);
}

BOOST_AUTO_TEST_CASE(TestLidDrivenCavity2)
{
    DefaultLidDrivenCavity dldc {};

    int worldSize {};
    int worldRank {};

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm grid;
    const int dims {2};
    const int nAxisRanks {sqrt(worldSize)};
    int sizes[dims] = {nAxisRanks, nAxisRanks};
    int periods[dims] = {0, 1};
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, dims, sizes, periods, reorder, &grid);

    int coords[dims];
    int gridRank {};

    MPI_Comm_rank(grid, &gridRank);
    MPI_Cart_coords(grid, gridRank, dims, coords);

    LidDrivenCavity solver {LidDrivenCavity()};

    int n {20*20};

    solver.SetDomainSize(dldc.Lx, dldc.Ly);
    solver.SetGridSize(20, 20);
    solver.SetTimeStep(0.005);
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
    std::string filePath = srcDir + "/data/default_solution_20_20_005.txt";

    DefaultLidDrivenCavityData data {n};
    ReadDefaultDataFromFile(filePath, &data, 20, 20);

    BOOST_TEST(ArraysAreEqual(u, data.u, n) == true);
    BOOST_TEST(ArraysAreEqual(v, data.v, n) == true);
    BOOST_TEST(ArraysAreEqual(vorticity, data.vorticity, n) == true);
    BOOST_TEST(ArraysAreEqual(streamFunction, data.streamFunction, n) == true);
}


BOOST_AUTO_TEST_CASE(TestLidDrivenCavity3)
{
    DefaultLidDrivenCavity dldc {};

    int worldSize {};
    int worldRank {};

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    MPI_Comm grid;
    const int dims {2};
    const int nAxisRanks {sqrt(worldSize)};
    int sizes[dims] = {nAxisRanks, nAxisRanks};
    int periods[dims] = {0, 1};
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, dims, sizes, periods, reorder, &grid);

    int coords[dims];
    int gridRank {};

    MPI_Comm_rank(grid, &gridRank);
    MPI_Cart_coords(grid, gridRank, dims, coords);

    LidDrivenCavity solver {LidDrivenCavity()};

    int n {10*10};

    solver.SetDomainSize(dldc.Lx, dldc.Ly);
    solver.SetGridSize(10, 10);
    solver.SetTimeStep(0.005);
    solver.SetFinalTime(dldc.T);
    solver.SetReynoldsNumber(dldc.Re);

    solver.Initialise();
    // solver.Integrate();

    // double u[n] {};
    // double v[n] {};

    // const double* const vorticity {solver.GetVorticity()};
    // const double* const streamFunction {solver.GetStreamFunction()};

    // solver.ConvertStreamFunctionToVelocityU(u);
    // solver.ConvertStreamFunctionToVelocityV(v);

    // std::string srcPath = std::experimental::filesystem::canonical(__FILE__).string();
    // std::string srcDir = srcPath.substr(0, srcPath.find_last_of("/\\"));
    // std::string filePath = srcDir + "/data/default_solution_10_10_005.txt";

    // DefaultLidDrivenCavityData data {n};
    // ReadDefaultDataFromFile(filePath, &data, 10, 10);

    // BOOST_TEST(ArraysAreEqual(u, data.u, n) == true);
    // BOOST_TEST(ArraysAreEqual(v, data.v, n) == true);
    // BOOST_TEST(ArraysAreEqual(vorticity, data.vorticity, n) == true);
    // BOOST_TEST(ArraysAreEqual(streamFunction, data.streamFunction, n) == true);
}
#include <iostream>
#include <mpi.h>
#include <cmath>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "../include/LidDrivenCavity.h"

int main(int argc, char **argv)
{
    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        cout << "Error initializing MPI." << endl;
        return EXIT_FAILURE;
    }
    
    int worldSize {};
    int worldRank {};

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (worldSize != 1 && worldSize != 4 && worldSize != 9 && worldSize != 16) {
        MPI_Finalize();
        std::cout << "Number if ranks not 1, 4, 9, or 16. Exiting program." << std::endl;
        return EXIT_FAILURE;
    }

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

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());
    solver->SetRank(coords[0], coords[1]);
    solver->SetSize(worldSize);
    solver->SetCommunicator(grid);

    // if (coords[0] == 0 && coords[1] == 0) {
    //     solver->PrintConfiguration();
    // }

    solver->Initialise();

    if (coords[0] == 0 && coords[1] == 0) {
        solver->WriteSolution("ic.txt");
    }
    solver->Integrate();
    
    if (coords[0] == 0 && coords[1] == 0) {
        solver->WriteSolution("final.txt");
    }

    MPI_Finalize();

	return 0;
}

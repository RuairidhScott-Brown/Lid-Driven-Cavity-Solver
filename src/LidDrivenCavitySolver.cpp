#include <iostream>
#include <mpi.h>
#include <cmath>
#include <omp.h>
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
        ("Nx",  po::value<int>()->default_value(201),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(201),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.005),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(0.5),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(1000),
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
    

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());

    if (worldRank == 0) {
        solver->PrintConfiguration();
    }

    solver->Initialise();

    if (worldRank == 0) {
        solver->WriteSolution("ic.txt");
    }
    solver->Integrate();
    
    if (worldRank == 0) {
        solver->WriteSolution("final.txt");
    }

    MPI_Finalize();

	return 0;
}

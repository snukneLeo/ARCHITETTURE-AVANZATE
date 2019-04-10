#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
	// Initialize the MPI environment
	MPI::Init(argc, argv);

	// Get the number of processes
	int world_size = MPI::COMM_WORLD.Get_size();

	// Get the rank of the process
	int world_rank = MPI::COMM_WORLD.Get_rank();

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI::Get_processor_name(processor_name, name_len);

	// Print off a hello world message
	printf("Hello world from processor %s, rank %d out of %d processors\n",
				processor_name, world_rank, world_size);

	// Finalize the MPI environment.
	MPI::Finalize();
}

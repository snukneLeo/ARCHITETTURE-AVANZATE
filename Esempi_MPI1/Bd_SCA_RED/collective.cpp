#include <cstdio>
#include <cstring>
#include <mpi.h>

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI::Init(argc, argv);

    // Get the number of processes
    int world_size = MPI::COMM_WORLD.Get_size();

    char buffer[] = "Ciao come va?";
    int count = std::strlen(buffer) + 1;

    // Get the rank of the process
    int world_rank = MPI::COMM_WORLD.Get_rank();

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI::Get_processor_name(processor_name, name_len);
    
    if (world_rank == 0) 
    {
        int source = 0;
        char buffer[] = "Ciao come va?";
        char buffer_rv[100];
        int count = std::strlen(buffer) + 1;
        //int tag = 0;

        //broadcast

        //invio in broadcast
        MPI::COMM_WORLD.Bcast(&world_rank,1,MPI::INT,source); //invio il mio rank
        

        //scatter
        MPI::COMM_WORLD.Scatter(&buffer,count,MPI::CHAR,&buffer_rv,count,MPI::CHAR,source);


        /* std::printf("Hello. this is %s: Message Received: \"%s\" of size %d from %s\n",
                    processor_name, buffer_rv, count_rv, processor_name);*/
    
    }
    else
    {
        int source = 0;
        //buffer rec
        int buffer_rv[100];

        //reduce
        MPI::COMM_WORLD.Reduce(&world_rank,&buffer_rv,count,MPI::INT,MPI::SUM,source);

        MPI::COMM_WORLD.Scatter(&buffer,1,MPI::CHAR,&buffer_rv,count,MPI::CHAR,source);


        // Print off a hello world message
        std::printf("Hello. this is %s: Message Received: \"%s\" of size %d from %s\n",
                    processor_name, buffer_rv, count, processor_name);
    }
    
    // Finalize the MPI environment.
    MPI::Finalize();
}

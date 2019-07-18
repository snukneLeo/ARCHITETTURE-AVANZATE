#include <cstdio>
#include <cstring>
#include <mpi.h>

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI::Init(argc, argv);

    //buffer
    char buffer[] = "Ciao stronzi! From 0";

    // Get the number of processes
    int world_size = MPI::COMM_WORLD.Get_size();

    // Get the rank of the process
    int world_rank = MPI::COMM_WORLD.Get_rank();

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI::Get_processor_name(processor_name, name_len);
    
    if (world_rank == 0) {
        int count = 3;//std::strlen(buffer) + 1;
        int source = 0;
        //buffer rec
        //char buffer_rv[100];

        int buffer_rv[MPI::COMM_WORLD.Get_size()];

        int count_rv = 21;

        //int tag = 0;

        //broadcast
        //MPI::COMM_WORLD.Bcast(buffer,count,MPI::CHAR,source);
        
        //scatter
        //MPI::COMM_WORLD.Scatter(buffer,count,MPI::CHAR,buffer_rv,count,MPI::CHAR,source);
        //gather
        //MPI::COMM_WORLD.Gather(&world_rank,1,MPI::INT,buffer_rv,world_size,MPI::INT,source);
        //reduce
        MPI::COMM_WORLD.Reduce(&world_rank,buffer_rv,1,MPI::INT,MPI::SUM,source);

        // Print off a hello world message

        for(int i = 0; i<sizeof(buffer_rv); i++)
            printf("%i\n",buffer_rv[i]);


        /* std::printf("Hello. this is %s: Message Received: \"%s\" of size %d from %s\n",
                    processor_name, buffer_rv, count_rv, processor_name);*/
    
    }
    else
    {
        int source = 0;
        //buffer rec
        int buffer_rv[100];
        int count = 3;//std::strlen(buffer) + 1; //DIMENSIONE PERFETTA
        
        //MPI::COMM_WORLD.Bcast(buffer, count, MPI::INT, source);

        //scatter rec
        //MPI::COMM_WORLD.Scatter(nullptr,count,MPI::CHAR,buffer_rv,count,MPI::CHAR,source);

        //gather
        //MPI::COMM_WORLD.Gather(&world_rank,1,MPI::INT,nullptr,1,MPI::INT,source);
        //reduce
        MPI::COMM_WORLD.Reduce(&world_rank,nullptr,1,MPI::INT,MPI::SUM,source);

        // Print off a hello world message
        /*std::printf("Hello. this is %s: Message Received: \"%s\" of size %d from %s\n",
                    processor_name, buffer_rv, count, processor_name);*/
    }
    
    // Finalize the MPI environment.
    MPI::Finalize();
}

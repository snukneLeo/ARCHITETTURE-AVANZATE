#include <cstdio>
#include <cstring>
#include <mpi.h>

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI::Init(argc, argv);

    int B [2][3];
    MPI::Datatype Matrix = MPI::INT.Create_contiguous(6);


    for(int i = 0; i<2; i++)
    {
        for (int j = 0; j<3; j++)
        {
            B[i][j] = i*j + 1*j;
        } 
    }


    for(int i = 0; i<2; i++)
    {
        for (int j = 0; j<3; j++)
        {
            printf("%i ",B[i][j]);
        } 
    }

    // Get the number of processes
    int world_size = MPI::COMM_WORLD.Get_size();

    // Get the rank of the process
    int world_rank = MPI::COMM_WORLD.Get_rank();

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI::Get_processor_name(processor_name, name_len);
    
    if (world_rank == 0) 
    {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI::Get_processor_name(processor_name, name_len);
        name_len++;

        std::printf("Hello. This is %s. I'm sending the message..\n",
                    processor_name);

        //char buffer[] = "Hello! from rank 0";
        //int count = std::strlen(buffer) + 1;
        int dest = 1;
        

        int tag = 0;
        MPI::COMM_WORLD.Send(B, 1, Matrix, dest, tag);

    }
    else if (world_rank == 1) {
        char buffer[100];
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int count, name_len;
        int source = 0;

        int tag = 0;
        MPI::COMM_WORLD.Recv(B, 1, Matrix, source, tag);

        // Print off a hello world message
        for(int i = 0; i<2; i++)
        {
            for (int j = 0; j<3; j++)
            {
                std::printf("Hello. this is %s: Message Received: \"%d\" of size %d from %s\n",
                    processor_name, B[i][j], count, processor_name);
            } 
        }
    }
    else
    std::printf("Hello. this is %s: I'm not playing... \n",
                processor_name);
    
    // Finalize the MPI environment.
    MPI::Finalize();
}

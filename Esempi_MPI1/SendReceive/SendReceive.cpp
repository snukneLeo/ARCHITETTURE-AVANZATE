#include <cstdio>
#include <cstring>
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
    
    if (world_rank == 0) {
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI::Get_processor_name(processor_name, name_len);
        name_len++;

        std::printf("Hello. This is %s. I'm sending the message..\n",
                    processor_name);
        
        //int chisono = MPI::COMM_WORLD.Get_rank();

        printf("Chi sono io: %i \n",world_rank);

        char buffer[] = "Hello! from rank 0";
        int count = std::strlen(buffer) + 1;
        int dest = 1;
        

        int tag = 0;
        MPI::COMM_WORLD.Send(&count, 1, MPI::INT, dest, tag);
        tag++;
        MPI::COMM_WORLD.Send(buffer, count, MPI::CHAR, dest, tag);
        tag++;
        MPI::COMM_WORLD.Send(&name_len, 1, MPI::INT, dest, tag);
        tag++;
        MPI::COMM_WORLD.Send(processor_name, name_len, MPI::CHAR, dest, tag);
        tag++;
        MPI::COMM_WORLD.Send("ciao franco F", 14, MPI::CHAR,dest,tag);

    }
    else if (world_rank == 1) {
        char buffer[100];
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int count, name_len;
        int source = 0;
        char buffer2[100];

        int tag = 0;
        MPI::COMM_WORLD.Recv(&count, 1, MPI::INT, source, tag);
        tag++;
        MPI::COMM_WORLD.Recv(buffer, count, MPI::CHAR, source, tag);
        tag++;
        MPI::COMM_WORLD.Recv(&name_len, 1, MPI::INT, source, tag);
        tag++;
        MPI::COMM_WORLD.Recv(processor_name, name_len, MPI::CHAR, source, tag);
        tag++;
        MPI::COMM_WORLD.Recv(buffer2,14,MPI::CHAR,source,tag);

        // Print off a hello world message
        std::printf("Hello. this is %s: Message Received: \"%s\" of size %d from %s\n",
                    processor_name, buffer, count, processor_name);
        // Print off a hello world message
        std::printf("Hello. this is %s: Message Received: \"%s\" of size %d from %s\n",
                    processor_name, buffer2, count, processor_name);
    }
    else
    std::printf("Hello. this is %s: I'm not playing... \n",
                processor_name);
    
    // Finalize the MPI environment.
    MPI::Finalize();
}

#include <cstdio>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI::Init(argc,argv); //set env

    //get the number of processes
    int worldSize = MPI::COMM_WORLD.Get_size();

    //get the rank of the processes
    int worldRank = MPI::COMM_WORLD.Get_rank();

    //get the name of the processor
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int nameLen = 0;

    //set tools
    int tag = 0;
    char buffer[] = "Hello!";
    int destinatario = 1; //successivo allo zero
    int mittente = 0; //initial state
    int count = std::strlen(buffer) + 1; //lunghezza messaggio

    //MPI::Get_processor_name(processorName,nameLen);

    if (worldRank == 0) //initial rank
    {       
        MPI::COMM_WORLD.Send(buffer,count, MPI::CHAR,destinatario,tag);
        destinatario =  (worldRank + 1)%4;
        //printf("SEND---Message: %s, Lunghezza: %d, Da chi: %d, A chi: %d, tag: %i\n",buffer,count,mittente,destinatario,tag);
        printf("SEND: FROM %d TO %d\n",mittente,destinatario);
    }
    else
    {
        mittente = worldRank;
        //ricevo il messaggio
        MPI::COMM_WORLD.Recv(buffer, count, MPI::CHAR, (mittente-1)%4, tag);
        //printf("REC---Message: %s, Lunghezza: %d, Da chi: %d, A chi: %d, tag: %i\n",buffer,count,mittente,destinatario,tag);
        printf("REC1: FROM %d\n",(mittente-1)%4);
        //count = std::strlen(buffer) + 1; //lunghezza messaggio
        destinatario = (worldRank + 1)%4;
        MPI::COMM_WORLD.Send(buffer,count, MPI::CHAR,destinatario,tag);
        //printf("SEND---Message: %s, Lunghezza: %d, Da chi: %d, A chi: %d, tag: %i\n",buffer,count,mittente,destinatario,tag);
        printf("SEND2: FROM %d TO %d\n",mittente,destinatario);
    }
    
    MPI::Finalize();
}
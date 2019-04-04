#include <iostream>
#include <omp.h>
#include "Timer.hpp"

int main() {
    using namespace timer;
    int N = (1 << 30);
    float tempSeq = 0, tempPar = 0;
    double factorial = 1;
    
    /* Sequential implementation of factorial:*/
    
    Timer<HOST> TM;
    TM.start();
    
    for (int i = 1; i <= N; ++i)
        factorial *= i;
    
    TM.stop();
    tempSeq = TM.duration();
    TM.print("Sequential Factorial");
    std::cout << factorial << std::endl;
    
    //--------------------------------------------------------------------------
    
    /* Parallel implementation of Factorial: */
    
    /*double parallelResult = 1;
    
    double* arrayfactorial = new double[omp_get_num_procs()]; //array con numero di thread del tuo piccione

    for (int i = 0; i<omp_get_num_procs(); i++)
        arrayfactorial[i] = 1;
    
    TM.start();

    #pragma omp parallel for firstprivate(parallelResult) //partiamo da 1 e non a caso come dice elia
    for (int i = 1; i<=N; ++i)
        arrayfactorial[omp_get_thread_num()] *= i; //calcolo per ogni cella corrispondente al thread il valore
    
    for (int i = 0; i<omp_get_num_procs(); ++i)
        parallelResult *= arrayfactorial[i];*/


    double parallelResult = 1;
    double partialFactorial = 1;

    TM.start();
    #pragma omp parallel firstprivate(partialFactorial) //creo una regione parallela di 8 thread
    {
        //imposto la varibile locale ad ogni thread
        // le thread ora si calcolano il risultato parziale
        // non creo però una regione parallela ulteriore
        // ma in questo caso uso le thread create prima
        #pragma omp for 
        for(int i = 1; i<=N; ++i)
            partialFactorial *= i;
        // avrà race condition sull'ultimo calcolo
        // mi rallenta leggermente ma non mi cambia 
        // nulla il risultato finale
        #pragma omp critical
        parallelResult *= partialFactorial;
    }
    
    //ottengo speed-up ~= 7
    TM.stop();
    tempPar = TM.duration();
    TM.print("Parallel Factorial");
    std::cout << parallelResult << std::endl;
    std::cout << "Speed-UP: " << std::endl;
    std::cout << tempSeq/tempPar << std::endl;
    return 0;
}

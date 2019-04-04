#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <rc4.hpp>
#include <Timer.hpp>

int main()
{
    using namespace timer;
    float tempSeq = 0, tempPar = 0;
    Timer<HOST> TM;
    TM.start();
    //rc4();
    TM.stop();
    tempSeq = TM.duration();
    std::cout << "done!" << std::endl<< std::endl;
    TM.print("Sequential RC4"); //stampo in secondi quanto ci ha messo

    TM.start();
    rc4_p();
    TM.stop();
    tempPar = TM.duration();
    std::cout << "done!" << std::endl<< std::endl;
    TM.print("Parallel RC4");
    std::cout << "Speed-UP: " <<std::endl;
    std::cout << tempSeq/tempPar << std::endl;
}
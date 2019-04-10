#include <iostream>
#include <omp.h>
#include <Timer.hpp>


long long int fibonacci(long long int value, int level) {
    if (value <= 1)
        return 1;

    long long int fib_left, fib_right;
    fib_left  = fibonacci(value - 1, level + 1);
    fib_right = fibonacci(value - 2, level + 1);

    return fib_left + fib_right;
}

// PARALLEL
long long int parallelFibonacci(long long int value, int level) 
{
    if (value <= 1)
        return 1;
    long long int fib_left = 1, fib_right = 1;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            fib_left  = parallelFibonacci(value - 1, level + 1);
            #pragma omp task
            fib_right = parallelFibonacci(value - 2, level + 1);
        }
    }

    return fib_left + fib_right;
}



int main() {
    using namespace timer;
    //  ------------------------- TEST FIBONACCI ----------------------
    omp_set_dynamic(0);
    int value = 40;
    Timer<HOST> TM;
    TM.start();
    long long int result = fibonacci(value, 1);
    TM.stop();
    TM.print("Seq time\n");
    

    TM.start();
    long long int parallelResult = parallelFibonacci(value,1);
    TM.stop();
    TM.print("Par time\n");

    std::cout << "\nresult: " << result << "\n" << std::endl;
    std::cout << "\nresult: " << parallelResult << "\n" << std::endl;
}

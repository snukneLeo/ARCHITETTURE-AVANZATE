mpicxx TestDerivated.cpp -std=c++0x -o testderivated
echo 'execute programm...'
mpirun testderivated 4

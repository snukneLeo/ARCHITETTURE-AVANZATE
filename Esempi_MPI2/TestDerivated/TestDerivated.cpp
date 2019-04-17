#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <algorithm>
#include <random>
#include <sstream>

int main(int argc, char* argv[]) {
	MPI::Init(argc, argv);
	MPI::COMM_WORLD.Set_errhandler(MPI::ERRORS_THROW_EXCEPTIONS);

	int n_of_tasks = MPI::COMM_WORLD.Get_size();
	int rank = MPI::COMM_WORLD.Get_rank(); //ogni processo ha il suo id da 0 a 3

	if (n_of_tasks != 4) {
		std::cout << "\n error set -n 4\n";
		return 0;
	}
	
	int name_len;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI::Get_processor_name(processor_name, name_len);

	// ------------------- TEST CONTINUOUS -----------------------------------------
	const int root = 0;
	int matrix3D[4][2][2];
	try {
		MPI::Datatype MatrixType = MPI::INT.Create_contiguous(4);
		MatrixType.Commit();
		
		if (rank == root) {
			for (int i = 0; i < n_of_tasks; ++i) {
				for (int j = 0; j < 2; ++j)
					for (int k = 0; k < 2; ++k)
						matrix3D[i][j][k] = i * j + k;
			}
			std::cout << "\nTEST CONTINUOUS\n" << std::endl;
		}
		int matrix2D[2][2]; //tutti i processi hanno la sua matrice 2D
		//invia i dati che deve mandare un processo che manda tutti i dati
		MPI::COMM_WORLD.Scatter(matrix3D, 1, MatrixType, matrix2D, 1, MatrixType, root); //invia il root
		//invio i 4 interi contigui 
		
		printf("rank %d -> Received Matrix {{%d, %d}, {%d, %d}}\n", rank, 
					matrix2D[0][0], matrix2D[0][1], matrix2D[1][0], matrix2D[1][1]);

	} catch (MPI::Exception e) {
		std::cout << "MPI ERROR: " << e.Get_error_code()
				  << " -" << e.Get_error_string() << std::endl;
	}
	
	// ------------------- TEST VECTOR -----------------------------------------
	
	MPI::COMM_WORLD.Barrier();
	
	int matrix[4][4];
	try {
		MPI::Datatype columnType = MPI::INT.Create_vector(4, 1, 4);
		columnType.Commit();
		
		if (rank == root) {
			for (int i = 0; i < n_of_tasks; ++i) {
				for (int j = 0; j < 4; ++j)
						matrix[i][j] = i * n_of_tasks + j;
			}
			std::cout << "\nTEST VECTOR\n" << std::endl;
			
			for (int i = 1; i < n_of_tasks; ++i)
				MPI::COMM_WORLD.Send(&matrix[0][i], 1, columnType, i, 0);
		}
		else {
			int recvColumn[4 * 4];
			MPI::COMM_WORLD.Recv(recvColumn, 1, columnType, root, 0);
		
			printf("rank %d -> Received Vector {%d, %d, %d, %d}\n", rank, 
						recvColumn[0], recvColumn[4], recvColumn[8], recvColumn[12]);
		}
	} catch (MPI::Exception e) {
		std::cout << "MPI ERROR: " << e.Get_error_code()
				  << " -" << e.Get_error_string() << std::endl;
	}
	
	// ------------------- TEST INDEXED -----------------------------------------
		
	try {
		MPI::COMM_WORLD.Barrier();
		int array_of_blocklengths[2] = {4, 2};
		int array_of_displacements[2] = {2, 12};
		MPI::Datatype MyIndexedType = MPI::INT.Create_indexed(2, array_of_blocklengths, array_of_displacements);
		MyIndexedType.Commit();
		
		int* array = new int[15]();
		if (rank == root) {
			for (int i = 0; i < 15; ++i)
				array[i] = i;
				
			std::cout << "\nTEST INDEXED\n" << std::endl;
		}
		MPI::COMM_WORLD.Bcast(array, 1, MyIndexedType, root);
		
		std::stringstream str;
		for (int i = 0; i < 15; ++i)
			str << array[i] << ' ';
		str << std::endl;
		
		std::cout << str.str();
	} catch (MPI::Exception e) {
		std::cout << "MPI ERROR: " << e.Get_error_code()
				  << " -" << e.Get_error_string() << std::endl;
	}
	
	// ------------------- TEST STRUCT -----------------------------------------
	
	MPI::COMM_WORLD.Barrier();
	
	struct {
		int num;
		float x;
		double data[4];		
	} obj;
	
	try {
		int blocks[3] = {1, 1, 4};
		MPI::Datatype types[3] = {MPI::INT, MPI::FLOAT, MPI::DOUBLE};

		MPI::Aint intExtent, floatExtent;
		MPI::Aint intLowerbound, floatLowerbound;
		MPI::INT.Get_extent(intLowerbound, intExtent);
		MPI::FLOAT.Get_extent(floatLowerbound, floatExtent);

		MPI::Aint displacements[3];
		displacements[0] = static_cast<MPI::Aint>(0);
		displacements[1] = intExtent;
		displacements[2] = intExtent + floatExtent;

		MPI::Datatype obj_type = MPI::Datatype::Create_struct(3, blocks, displacements, types);
		obj_type.Commit();
			
	MPI::COMM_WORLD.Barrier();
		if (rank == root) {
			obj.num = 3;
			obj.x = 4.56;
			for (int i = 0; i < 4; ++i)
				obj.data[i] = i + 0.5;
				
			std::cout << "\nTEST STRUCT\n" << std::endl;
			
		}
		MPI::COMM_WORLD.Bcast(&obj, 1, obj_type, root);
		
		std::stringstream str;
		str << obj.num << '\t' << obj.x << '\t';
		for (int i = 0; i < 4; ++i)
			str << obj.data[i] << "  ";
		str << std::endl;
		
		std::cout << str.str();
	} catch (MPI::Exception e) {
		std::cout << "MPI ERROR: " << e.Get_error_code()
				  << " -" << e.Get_error_string() << std::endl;
	}

	MPI::Finalize();
}

//@author Federico Busato
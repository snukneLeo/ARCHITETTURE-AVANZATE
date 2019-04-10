#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <Timer.hpp>

void test_producer_consumer(int Buffer[32])  //SEQUENTIAL
{
	int i = 0;
	int count = 0;

	while (i < 35000) {					// number of test

		// PRODUCER
		if ((rand() % 50) == 0) {		// some random computations

			if (count < 31) {
				++count;
				// std::cout << "Thread:\t" << omp_get_thread_num()
                //           << "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations

			if (count >= 1) {
				int var = Buffer[count];
				// std::cout << "Thread:\t" << omp_get_thread_num()
                //           << "\tConsume on index: " << count
                //           << "\tvalue: " << var << std::endl;
				--count;
			}
		}
		i++;
	}
}


void test_producer_consumer_parallel_for(int Buffer[32])  //FOR
{
	int i = 0;
	int count = 0;

	#pragma omp parallel
	for (i = 0; i < 35000; i++)
	{					
		if ((rand() % 50) == 0) 
		{		// some random computations
			if (count < 31) 
			{
				//#pragma omp atomic
				++count;
				// std::cout << "Thread:\t" << omp_get_thread_num()
				// 		<< "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
		}

		// CONSUMER
		if (omp_get_thread_num() != 0)
		{
			if ((std::rand() % 51) == 0) 
			{		// some random computations

				if (count >= 1) 
				{
					int var = Buffer[count];
					// std::cout << "Thread:\t" << omp_get_thread_num()
					// 	<< "\tConsume on index: " << count
					// 	<< "\tvalue: " << var << std::endl;
					//#pragma omp atomic
					--count;
				}
			}
		}
	}
}

void test_producer_consumer_p_CRITICAL(int Buffer[32])  //CRITICAL
{
	int i = 0;
	int count = 0;
	#pragma omp parallel
	for (i = 0; i < 35000; i++)
	{					
		// PRODUCER
		if ((rand() % 50) == 0) 
		{		// some random computations

			#pragma omp critical
			{
				#pragma omp flush(count)
				if (count < 31)
				{
					++count;
					// std::cout << "Thread:\t" << omp_get_thread_num()
					// 	<< "\tProduce on index: " << count << std::endl;
					Buffer[count] = omp_get_thread_num();
				}
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) 
		{		// some random computations
			#pragma omp critical
			{
				if (count >= 1) 
				{
					#pragma omp flush(count)
					int var = Buffer[count];
					// std::cout << "Thread:\t" << omp_get_thread_num()
					// 	<< "\tConsume on index: " << count
					// 	<< "\tvalue: " << var << std::endl;
					--count;
				}
			}
		}
	}
}


void test_producer_consumer_p_LOCK(int Buffer[32])  //LOCK
{
	int i = 0;
	int count = 0;
	omp_lock_t lock;
	omp_init_lock(&lock);
	#pragma omp parallel
	for (i = 0; i < 35000; i++)
	{					
		// PRODUCER
		if ((rand() % 50) == 0) 
		{		// some random computations
			omp_set_lock(&lock); //lo prendo
			if (count < 31) 
			{
				++count;
				// std::cout << "Thread:\t" << omp_get_thread_num()
				// 		<< "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
			omp_unset_lock(&lock);
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) {		// some random computations

			omp_set_lock(&lock); //lo prendo
			if (count >= 1) 
			{
				int var = Buffer[count];
				// std::cout << "Thread:\t" << omp_get_thread_num()
				// 		<< "\tConsume on index: " << count
				// 		<< "\tvalue: " << var << std::endl;
				--count;
			}
			omp_unset_lock(&lock);
		}
	}
	omp_destroy_lock(&lock);
}


void test_producer_consumer_p_BARRIER(int Buffer[32])  //BARRIER
{
	int i = 0;
	int count = 0;
	#pragma omp parallel
	{
		#pragma omp schedule(static)
		for (i = 0; i < 35000; i++)
		{					
			// PRODUCER
			if ((rand() % 50) == 0) 
			{		// some random computations
				if (count < 31) 
				{
					#pragma omp barrier
					++count;
					// std::cout << "Thread:\t" << omp_get_thread_num()
					// 		<< "\tProduce on index: " << count << std::endl;
					Buffer[count] = omp_get_thread_num();
				}
			}

			// CONSUMER
			if ((std::rand() % 51) == 0) 
			{		// some random computations
				if (count >= 1) 
				{
					int var = Buffer[count];
					// std::cout << "Thread:\t" << omp_get_thread_num()
					// 		<< "\tConsume on index: " << count
					// 		<< "\tvalue: " << var << std::endl;
					--count;
					#pragma omp barrier
				}
			}
		}
	}
}


void test_producer_consumer_parallel_ATOMIC(int Buffer[32])  //ATOMIC
{
	int i = 0;
	int count = 0;

	#pragma omp parallel
	for (i = 0; i < 35000; i++)
	{					
		// PRODUCER
		if ((rand() % 50) == 0) 
		{		// some random computations
			if (count < 31) 
			{
				//#pragma omp atomic
				++count;
				// std::cout << "Thread:\t" << omp_get_thread_num()
				// 		<< "\tProduce on index: " << count << std::endl;
				Buffer[count] = omp_get_thread_num();
			}
		}

		// CONSUMER
		if ((std::rand() % 51) == 0) 
		{		// some random computations

			if (count >= 1) 
			{
				int var = Buffer[count];
				// std::cout << "Thread:\t" << omp_get_thread_num()
				// 	<< "\tConsume on index: " << count
				// 	<< "\tvalue: " << var << std::endl;
				//#pragma omp atomic
				--count;
			}
		}
	}
}

int main() 
{
	using namespace timer;

	int Buffer[32];
	std::srand(time(NULL));
	omp_set_num_threads(2);

	Timer<HOST> TM;

	TM.start();
	test_producer_consumer(Buffer);
	TM.stop();
	TM.print("SEQUENTIAL:\t");

	TM.start();
	test_producer_consumer_parallel_for(Buffer);
	TM.stop();
	TM.print("PAPARALLEL FOR:\t");

	TM.start();
	test_producer_consumer_p_CRITICAL(Buffer);
	TM.stop();
	TM.print("CRITICAL:\t");

	TM.start();
	test_producer_consumer_p_LOCK(Buffer);
	TM.stop();
	TM.print("LOCK:\t");

	TM.start();
	test_producer_consumer_p_BARRIER(Buffer);
	TM.stop();
	TM.print("BARRIER:\t");

	TM.start();
	test_producer_consumer_parallel_ATOMIC(Buffer);
	TM.stop();
	TM.print("ATOMIC:\t");
}

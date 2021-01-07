#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <chrono>

//Uncomment printmat and use low matrix values (multiple of task count) to compare matrix results, seemed accurate

//Approximate Time Taken with 4 Tasks:
//Serial: 17.455s
//Send / Receive: 6.449s
//Collective: 5.357s

//Approximate Time Taken with 2 Tasks:
//Serial: 17.455s
//Send / Receive: 17.985s
//Collective: 9.135s

//Approximate Time Taken with 1 Tasks:
//Serial: 17.455s
//Send / Receive: 17.985s
//Collective: 18.131s

// ---------------------------------------------------------------------------
// allocate space for empty matrix A[row][col]
// access to matrix elements possible with:
// - A[row][col]
// - A[0][row*col]

float** alloc_mat(int row, int col)
{
	float** A1, * A2;

	A1 = (float**)calloc(row, sizeof(float*));	 // pointer on rows
	A2 = (float*)calloc(row * col, sizeof(float));    // all matrix elements
	for (int i = 0; i < row; i++)
		A1[i] = A2 + i * col;

	return A1;
}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float** A, int row, int col)
{
	for (int i = 0; i < row * col; i++)
		A[0][i] = (float)(rand() % 10);
}

// ---------------------------------------------------------------------------
// DEBUG FUNCTION: printout of all matrix elements

void print_mat(float** A, int row, int col, char const* tag)
{
	int i, j;

	printf("Matrix %s:\n", tag);
	for (i = 0; i < row; i++)
	{
		for (j = 0; j < col; j++)
			printf("%6.1f   ", A[i][j]);
		printf("\n");
	}
}

// ---------------------------------------------------------------------------
// free dynamically allocated memory, which was used to store a 2D matrix
void free_mat(float** A, int num_rows) {
	free(A[0]); // free contiguous block of float elements (row*col floats)
	free(A);    // free memory for pointers pointing to the beginning of each row
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	int nodeID, numNodes;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);
	MPI_Status status;

	/* print user instruction */
	if (argc != 4)
	{
		printf("Matrix multiplication: C = A x B\n");
		printf("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]);
		return 0;
	}

	int d1, d2, d3;         // dimensions of matrices
	int i, j, k;			// loop variables

	/* read user input */
	d1 = atoi(argv[1]);		// rows of A and C
	d2 = atoi(argv[2]);     // cols of A and rows of B
	d3 = atoi(argv[3]);     // cols of B and C

	float** A, ** B;

	A = alloc_mat(d1, d2);
	init_mat(A, d1, d2);
	B = alloc_mat(d2, d3);
	init_mat(B, d2, d3);

	//parallel send / receive version
	{
		std::chrono::milliseconds start, end;

		start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		float ** C;	// matriCes

		printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2, d2, d3);

		/* prepare matrices */
		C = alloc_mat(d1, d3);	// no initialisation of C, because it gets filled by matmult

		//float sendBuf[1];
		//float recvBuf[1];

		float* sendBuf = (float*)calloc(d2 > d3 ? d2 : d3, sizeof(float));
		float* recvBuf = (float*)calloc(d2 > d3 ? d2 : d3, sizeof(float));

		if (0 == nodeID)
		{
			printf("Sending to other tasks\n\n");
			{
				std::chrono::milliseconds start, end;

				start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

				for (int i = 1; i < numNodes; ++i)
				{
					for (int j = 0; j < d2; ++j)
					{
						sendBuf[j] = A[d1 * (i - 1) / (numNodes - 1)][j];
					}
					MPI_Send(sendBuf, d2, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
				}
				
				end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

				printf("\nParallel Send / Receive Time Taken To Send in Milliseconds: %lld\n\n\n", end.count() - start.count());
			}

			printf("Receiving results from other tasks\n\n");
			for (int i = 1; i < numNodes; ++i)
				for (int i2 = d1 * (nodeID - 1) / (numNodes - 1); i2 < (d1 * nodeID / (numNodes - 1)); i2++)
				{
					MPI_Recv(recvBuf, d3, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);

					for (int j = 0; j < d3; ++j)
					{
						C[d1 * (i - 1) / (numNodes - 1)][j] += recvBuf[j];
					}
				}

			/*print_mat(A, d1, d2, "A");
			print_mat(B, d2, d3, "B");
			print_mat(C, d1, d3, "C");*/
		}
		else
		{
			//Calculation Part
			{
				std::chrono::milliseconds start, end;

				MPI_Recv(recvBuf, d2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
				for (int j = 0; j < d2; ++j)
				{
					A[0][j] = recvBuf[j];
				}
				//print_mat(A, d1, d2, "A");
				start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

				printf("Node %d: performing parallel matrix multiplication...\n", nodeID);
				//for (i = d1 * (nodeID - 1) / (numNodes - 1); i < (d1 * nodeID / (numNodes - 1)); i++)
				for (i = 0; i < d1 / (numNodes - 1); ++i)
					for (j = 0; j < d3; j++)
						for (k = 0; k < d2; k++)
							C[i][j] += A[0][k] * B[k][j];

				end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

				printf("\nParallel Send / Receive Calculation Time Taken in Milliseconds: %lld\n\n\n", end.count() - start.count());
			}

			//Return Part
			{
				std::chrono::milliseconds start, end;

				start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

				for (i = 0; i < d1 / (numNodes - 1); ++i)
				{
					for (j = 0; j < d3; j++)
					{
						sendBuf[j] = C[i][j];
					}

					MPI_Send(sendBuf, d3, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
				}

				end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

				printf("\nParallel Send / Receive Time Taken To Receive (Return Send) in Milliseconds: %lld\n\n\n", end.count() - start.count());
			}
		}

		end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("\nParallel Send / Receive Total Time Taken in Milliseconds: %lld\n\n\n", end.count() - start.count());
	}

	//parallel collective version
	{
		std::chrono::milliseconds start, end;

		start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		for (k = 0; k < d2; k++) MPI_Bcast(B[k], d3, MPI_FLOAT, 0, MPI_COMM_WORLD);

		float** C = alloc_mat(d1, d3);

		float** rowsA = alloc_mat(d1 / numNodes, d2);
		float** rowsC = alloc_mat(d1 / numNodes, d3);

		{
			std::chrono::milliseconds start, end;

			start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

			for (i = 0; i < d1 / numNodes; ++i)
				MPI_Scatter(A[i], d2, MPI_FLOAT, rowsA[i], d2, MPI_FLOAT, 0, MPI_COMM_WORLD);

			end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

			printf("\nParallel Collective Time Taken To Scatter in Milliseconds: %lld\n", end.count() - start.count());
		}

		{
			std::chrono::milliseconds start, end;

			start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

			for (i = 0; i < d1 / numNodes; ++i)
				for (j = 0; j < d3; j++)
					for (k = 0; k < d2; k++)
						rowsC[i][j] += rowsA[i][k] * B[k][j];

			end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

			printf("\nParallel Collective Time Taken To Calculate in Milliseconds: %lld\n", end.count() - start.count());
		}

		
		{
			std::chrono::milliseconds start, end;

			start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

			for (i = 0; i < d1 / numNodes; ++i)
				MPI_Gather(rowsC[i], d3, MPI_FLOAT, C[i], d3, MPI_FLOAT, 0, MPI_COMM_WORLD);

			end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

			printf("\nParallel Collective Time Taken To Gather in Milliseconds: %lld\n", end.count() - start.count());
		}

		if (0 == nodeID)
		{
			/*print_mat(A, d1, d2, "A");
			print_mat(B, d2, d3, "B");
			print_mat(C, d1, d3, "C");*/
		}

		end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("\nParallel Collective Total Time Taken: %lld\n", end.count() - start.count());
	}

	/* serial version of matmult */
	if (0 == nodeID)
	{
		float** C;	// matrices

		printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2, d2, d3);

		/* prepare matrices */
		C = alloc_mat(d1, d3);	// no initialisation of C, because it gets filled by matmult

		std::chrono::milliseconds start, end;

		start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("Perform matrix multiplication...\n");
		for (i = 0; i < d1; i++)
			for (j = 0; j < d3; j++)
				for (k = 0; k < d2; k++)
					C[i][j] += A[i][k] * B[k][j];

		end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("\nSerial Time Taken in Milliseconds: %lld\n\n\n", end.count() - start.count());

		/* test output */
		/*print_mat(A, d1, d2, "A");
		print_mat(B, d2, d3, "B");
		print_mat(C, d1, d3, "C");*/

		printf("\nDone.\n");

		/* free dynamic memory */
		free_mat(A, d1);
		free_mat(B, d2);
		free_mat(C, d1);
	}


	MPI_Finalize();

	return 0;
}

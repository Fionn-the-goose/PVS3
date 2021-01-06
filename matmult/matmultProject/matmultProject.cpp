#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#include <chrono>

//#define GET_TIME std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())

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

	float** A, ** B, ** C;	// matrices
	int d1, d2, d3;         // dimensions of matrices
	int i, j, k;			// loop variables

	//serial stuff
	/* print user instruction */
	if (argc != 4)
	{
		printf("Matrix multiplication: C = A x B\n");
		printf("Usage: %s <NumRowA> <NumColA> <NumColB>\n", argv[0]);
		return 0;
	}

	/* read user input */
	d1 = atoi(argv[1]);		// rows of A and C
	d2 = atoi(argv[2]);     // cols of A and rows of B
	d3 = atoi(argv[3]);     // cols of B and C

	printf("Matrix sizes C[%d][%d] = A[%d][%d] x B[%d][%d]\n", d1, d3, d1, d2, d2, d3);

	/* prepare matrices */
	A = alloc_mat(d1, d2);
	init_mat(A, d1, d2);
	B = alloc_mat(d2, d3);
	init_mat(B, d2, d3);
	C = alloc_mat(d1, d3);	// no initialisation of C, because it gets filled by matmult

	//parallel send / receive version
	if (0 == nodeID)
	{
		printf("Receiving results from other tasks\n\n");
		for (int i = 1; i < numNodes; ++i)
		{
			MPI_Recv(C[d1 * (i - 1) / (numNodes - 1)], d1 / (numNodes - 1), MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
		}
	}
	else
	{
		std::chrono::milliseconds start, end;

		start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("Node %d: performing parallel matrix multiplication...\n", nodeID);
		//for (i = d1 * nodeID / numNodes; i < (d1 * (nodeID + 1) / numNodes); i++)
		for(i = 0; i < d1 / (numNodes - 1); ++i)
		{
			//printf("%d ", i);

			for (j = 0; j < d3; j++)
				for (k = 0; k < d2; k++)
					C[i][j] += A[i][k] * B[k][j];
		}

		MPI_Send(C[d1 * (nodeID - 1) / (numNodes - 1)], d1 / (numNodes - 1), MPI_FLOAT, 0, 1, MPI_COMM_WORLD);

		end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("\nParallel Send / Receive Time Taken in Milliseconds: %lld\n\n\n", end.count() - start.count());
	}

	/* serial version of matmult */
	if (0 == nodeID)
	{
		std::chrono::milliseconds start, end;

		start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("Perform matrix multiplication...\n");
		for (i = 0; i < d1; i++)
			for (j = 0; j < d3; j++)
				for (k = 0; k < d2; k++)
					C[i][j] += A[i][k] * B[k][j];

		end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

		printf("\nTime Taken in Milliseconds: %lld\n\n\n", end.count() - start.count());

		/* test output */
		//print_mat(A, d1, d2, "A");
		//print_mat(B, d2, d3, "B");
		//print_mat(C, d1, d3, "C");

		printf("\nDone.\n");

		/* free dynamic memory */
		free_mat(A, d1);
		free_mat(B, d2);
		free_mat(C, d1);
	}


	MPI_Finalize();

	return 0;
}

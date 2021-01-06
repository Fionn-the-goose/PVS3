#include<stdio.h>
#include<mpi.h>									// ???
 
int main(int argc, char** argv){
	int nodeID, numNodes;

	/* ??? */
	MPI_Init(&argc, &argv);						// ???
	MPI_Comm_size(MPI_COMM_WORLD, &numNodes);	// ???
	MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);		// ???

	/* ??? */
	printf("Hello world from process %d of %d\n", nodeID, numNodes);

	/* ??? */
	MPI_Finalize();

	return 0;
}

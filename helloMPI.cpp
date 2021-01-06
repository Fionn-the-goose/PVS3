#include<stdio.h>
#include<mpi.h>									// Bibliothek welche die funktionalitaet von MPI anbietet
 
int main(int argc, char** argv){
	int nodeID, numNodes;

	/* Startet die parallele Sektion, und intialisiert wichtige werte */
	MPI_Init(&argc, &argv);						// Initialisiert eine Sektion welche von alle unseren threads verwendet wird
	MPI_Comm_size(MPI_COMM_WORLD, &numNodes);	// Speichert die anzahl an threads an denen die arbeit aufgeteilt ist
	MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);		// Speichert den wert vom thread welcher momentan diesen code ausfuehrt

	/* Gibt information ueber momentanen thread aus */
	printf("Hello world from process %d of %d\n", nodeID, numNodes);

	/* Schliesst die parallele Sektion */
	MPI_Finalize();

	return 0;
}

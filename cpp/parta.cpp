
#include "mpi.h"
#include "mkl.h"
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

#define MASTERID 0
#define NPROC 4


void	LUSolver(double *L, double *U, double *f, int N,int n, int rank, int size){
	
	// solve Ly=f;
	for(int p=0;p<size;p++){
		if(rank == p){
			double *b=new double[N];
			double *temp=new double[n];
			cblas_dcopy(n,f+p*n,1,temp,1);
			cblas_dgemv (CblasColMajor, CblasNoTrans, N,  n, 1.0, L, N, temp, 1, 0.0, b,1); // b=L*Temp
			
			cblas_daxpy (N-((p+1)*n), -1.0,b+(p+1)*n, 1,f+(p+1)*n,1);
			
		}
		MPI_Bcast(f, N, MPI_DOUBLE, p, MPI_COMM_WORLD);
	}
		

	
	// solve Ux=y;
	for(int p=size-1;p>0;p--){
		if(rank == p){
			double *Uii=new double[n*n]; //col major
			for(int i=0;i<n;i++)
				for(int j=0;j<n;j++)
					*(Uii+i+j*n)=*(U+ (i+p*n)+ j*N);
			double *temp=new double[n];
			cblas_dcopy(n,f+p*n,1,temp,1);

			MKL_INT *ipiv=new MKL_INT[n];
			LAPACKE_dgetrf (LAPACK_COL_MAJOR, n , n , Uii , n ,  ipiv );
			LAPACKE_dgetrs (LAPACK_COL_MAJOR , 'N', n , 1, Uii , n  ,ipiv , temp ,n);
			
			double *b=new double[N];
			cblas_dgemv (CblasColMajor, CblasNoTrans, N,  n, 1.0, U, N, temp, 1, 0.0, b,1); // b=U*temp
			cblas_daxpy (p*n, -1.0,b, 1,f,1);
			cblas_dcopy(n,temp,1,f+p*n,1);

		}
		MPI_Bcast(f, N, MPI_DOUBLE, p, MPI_COMM_WORLD);
	}
	int p=0;
	// solve A11x1=f1
	if(rank == p){
			double *Uii=new double[n*n]; //col major
			for(int i=0;i<n;i++)
				for(int j=0;j<n;j++)
					*(Uii+i+j*n)=*(U+ (i+p*n)+ j*N);
			double *temp=new double[n];
			cblas_dcopy(n,f+p*n,1,temp,1);
		/*			
			for(int i=0;i<n;i++){
				for(int j=0;j<n;j++)
					cout<<*(Uii+i+j*n)<< " ";
				cout<<endl;	
			}
		*/
			MKL_INT *ipiv=new MKL_INT[n];
			LAPACKE_dgetrf (LAPACK_COL_MAJOR, n , n , Uii , n ,  ipiv );
			LAPACKE_dgetrs (LAPACK_COL_MAJOR , 'N', n , 1, Uii , n  ,ipiv , temp ,n);
			cblas_dcopy(n,temp,1,f+p*n,1);



		}
		MPI_Bcast(f, N, MPI_DOUBLE, p, MPI_COMM_WORLD);


}











int main(int argc, char** argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N, M, n;
	double tic,toc;
	double  *L, *U;
	double *L_local,*U_local;
	double *f, *b;

	if (rank == MASTERID) {

		/*------ Read dense matrix L in fileL argv[1]: ------*/
		ifstream fileL(argv[1]);
		// Ignore headers and comments:
		while (fileL.peek() == '%') {
			fileL.ignore(2048, '\n');
		}
		
		// Read defileAing parameters:
		fileL >> M >> N ;

		L=new double[M*N]; // L is col major		

		for (int i = 0; i < M*N; i++) {
			fileL >> *(L+i);
		}
		fileL.close();

		/*------ Read dense matrix U in fileU argv[1]: ------*/
		ifstream fileU(argv[2]);
		// Ignore headers and comments:
		while (fileU.peek() == '%') {
			fileU.ignore(2048, '\n');
		}
		
		// Read defileAing parameters:
		fileU >> M >> N ;

		U=new double[M*N]; // U is col major		

		for (int i = 0; i < M*N; i++) {
			fileU >> *(U+i);
		}
		fileU.close();
	}
	MPI_Bcast(&N, 1, MPI_INT, MASTERID, MPI_COMM_WORLD);
	f = new double[N];

	if (rank == MASTERID) {

		/*------ Read right hand side f in argv[3] ------*/
		ifstream filef(argv[3]);
		// Ignore headers and comments:
		while (filef.peek() == '%') {
			filef.ignore(2048, '\n');
		}
		// Read defileAing parameters:
		filef >> N;
		filef.ignore(2048, '\n');

		
		for (int i = 0; i < N; i++) {
			filef >> f[i];
		}
		filef.close();

		/*
		for(int i=0;i<N;i++)
			cout<<f[i]<<endl;
		*/

	}

	
	MPI_Bcast(f, N, MPI_DOUBLE, MASTERID, MPI_COMM_WORLD);
	M=N;
	n = N / size;

	L_local = new double[N*n];
	U_local = new double[N*n];
	MPI_Scatter(L, N*n, MPI_DOUBLE, L_local, N*n, MPI_DOUBLE,
			MASTERID, MPI_COMM_WORLD);
	MPI_Scatter(U, N*n, MPI_DOUBLE, U_local, N*n, MPI_DOUBLE,
			MASTERID, MPI_COMM_WORLD);
	
	/*
	if(rank==0){
		for(int i=0;i<N;i++){
			for (int j=0;j<n;j++)
				cout<<*(L_local+ i+j*N     )<<" ";
			cout<<endl;
			
		}
	}
	*/
		
	LUSolver(L_local,U_local,f,N,n, rank,size);

	
	if(rank==MASTERID){
		for(int i=0;i<N;i++){
			cout<<f[i]<<endl;
			
		}
	}
	


	MPI_Finalize();
}

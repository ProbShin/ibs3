/*************************************************************************
    > File Name: main.cpp
    > Author: xc
    > Descriptions: 
    > Created Time: Tue May 10 16:22:56 2016
 ************************************************************************/
#include "mpi.h"
#include "lab3.h"
#include <omp.h>
#include "mkl.h"
#include <string>
//#include<mkl_scalapack.h>
#include <unistd.h>  /*sleep*/
#include<assert.h>
#define NPROC 8
#define ROOT_ID 0
#define MASTERID 0

using namespace std;

void blockLU(double *L, double *U, int mm, int rank);
void	LUSolver(double *L, double *U, double *f, int N,int n, int rank, int size);


int main(int argc, char** argv){
  string s1 = argv[1];
  string s2 = argv[2];
  string s3 = argv[3];

  int size, rank, proc_name_len, rc;
  char procname[MPI_MAX_PROCESSOR_NAME];
  double runner_time=0;
  if( (rc=MPI_Init(&argc, &argv))!=MPI_SUCCESS){
    cerr<<"Error start MPI program!"<<endl;
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
  }  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double *arr_all=NULL;
  double *U = NULL;  //blk col domain
  double *L = NULL;  //blk col domain
  double *f = NULL;
  double *UU= NULL;  //full col domain
  double *LL= NULL;  //full col domain
  int mm;            //mm= size/NPROC

  if(rank == ROOT_ID) {
    mtxBLU mtxA = mtxBLU(s1.c_str(),NPROC);
    mm= mtxA.m/NPROC;
    arr_all = new double[mm*mm*NPROC*NPROC];
    for(int i=0; i<mm*mm*NPROC*NPROC; i++)
      arr_all[i] = (mtxA.arr)[i];

    mtxBLU mtxb = mtxBLU(s2.c_str());
    f = new double[mm*NPROC];
    for(int i=0; i<mm*NPROC; i++){
      f[i]=mtxb.arr[i];
    }
  }

  MPI_Bcast(&mm, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
  if(rank!=ROOT_ID)
    f=new double[mm*NPROC];
  MPI_Bcast(f , mm*NPROC, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
  U = new double[NPROC*mm*mm]; //block col domain
  L = new double[NPROC*mm*mm]; //block col domain
  UU= new double[NPROC*mm*mm]; //full col domain
  LL= new double[NPROC*mm*mm]; //full col domain
  std::fill(L, L+NPROC*mm*mm, 0);
  std::fill(U, U+NPROC*mm*mm, 0);
  for(int i=0; i<mm; i++)
    L[rank*mm*mm+i+i*mm]=1;  //make Lii = eyes(mm)

  MPI_Scatter(arr_all, mm*mm*8, MPI_DOUBLE,
              U      , mm*mm*8, MPI_DOUBLE,
              ROOT_ID, MPI_COMM_WORLD);

  blockLU(L, U, mm, rank);

  for(int k=0; k<NPROC; k++)
    for(int j=0; j<mm; j++)
      for(int i=0; i<mm; i++){
        UU[k*mm+i+j*mm*NPROC]= U[k*mm*mm+j*mm+i];
        LL[k*mm+i+j*mm*NPROC]= L[k*mm*mm+j*mm+i];
      }

  LUSolver(LL, UU, f, mm*NPROC, mm, rank, NPROC);


  if(rank==ROOT_ID)
    for(int i=0;i<mm*NPROC; i++)
      cout<<f[i]<<endl;

  delete [] f;
  delete [] arr_all;
  delete [] U;
  delete [] L;
  delete [] UU;
  delete [] LL;
  MPI_Finalize();
  return 0;
}



void blockLU(double *L, double *U, int mm, int rank){
  double *&arr = U;
  double *G    = new double[mm*mm];
  MKL_INT *ipiv= new MKL_INT[ mm ];
  std::fill(G   , G+mm*mm, 0);
  std::fill(ipiv, ipiv+mm, 0);

  for(int p=0; p<NPROC; p++){
    if(rank == p){
      //inv(App)
      LAPACKE_dgetrf (LAPACK_COL_MAJOR, mm, mm, U+p*mm*mm, mm, ipiv );
      LAPACKE_dgetri (LAPACK_COL_MAJOR, mm,     U+p*mm*mm, mm, ipiv );
    }
    for(int i=p+1; i<NPROC; i++){
      if(rank==p){
        //Gi stored into Li
        cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, mm, mm, mm, 1, U+i*mm*mm, mm, U+p*mm*mm, mm, 0, L+i*mm*mm, mm);
        MPI_Bcast(L+i*mm*mm, mm*mm, MPI_DOUBLE, p, MPI_COMM_WORLD);
        std::fill(U+i*mm*mm, U+i*mm*mm+mm*mm,0);
      }
      else if(rank > p){
        MPI_Bcast(G, mm*mm, MPI_DOUBLE, p, MPI_COMM_WORLD);
        MKL_INT j=i+rank-p;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mm, mm, mm, -1, G, mm, U+p*mm*mm, mm, 1, U+i*mm*mm, mm);//Aij-=G*Ajp
      }
      else{
        MPI_Bcast(G, mm*mm, MPI_DOUBLE, p, MPI_COMM_WORLD);
      }
    }//endof for i

  }//endof for p


  ////printf("@%d@\n",rank);
  ////MPI_Barrier(MPI_COMM_WORLD);
  ////if(rank == 4){
  ////  for(int k=0; k<NPROC; k++){
  ////    printf("The U%d == ",k);
  ////    for(int i=0;i<mm*mm; i++)
  ////      printf("%f ", U[k*mm*mm+i]);
  ////    printf("\n");
  ////  }
  ////
  ////  for(int k=0; k<NPROC; k++){
  ////    printf("The L%d == ",k);
  ////    for(int i=0;i<mm*mm; i++)
  ////      printf("%f ", L[k*mm*mm+i]);
  ////    printf("\n");
  ////   }
  //// sleep(5);
  //// }
  ////
  //// MPI_Barrier(MPI_COMM_WORLD);

  delete []G;
  delete []ipiv;
}//endof function








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

      //MKL_INT *ipiv=new MKL_INT[n];
      //LAPACKE_dgetrf (LAPACK_COL_MAJOR, n , n , Uii , n ,  ipiv );
      //LAPACKE_dgetrs (LAPACK_COL_MAJOR , 'N', n , 1, Uii , n  ,ipiv , temp ,n);
      cblas_dgemv (CblasColMajor,CblasNoTrans, n, n, 1, Uii, n, temp, 1, 0, temp, 1);


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
    //MKL_INT *ipiv=new MKL_INT[n];
    //LAPACKE_dgetrf (LAPACK_COL_MAJOR, n , n , Uii , n ,  ipiv );
    //LAPACKE_dgetrs (LAPACK_COL_MAJOR , 'N', n , 1, Uii , n  ,ipiv , temp ,n);

    cblas_dgemv (CblasColMajor,CblasNoTrans, n, n, 1, Uii, n, temp, 1, 0, temp, 1);
    cblas_dcopy(n,temp,1,f+p*n,1);



  }
  MPI_Bcast(f, N, MPI_DOUBLE, p, MPI_COMM_WORLD);


}








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


using namespace std;

void blockLU(int rank, string &s);

int inline pAij(int lda, int i, int j){
  return i+lda*j;
}


int main(int argc, char** argv){

  string f1 = argv[1];
  string f2 = argv[2];
  string f3 = argv[3];
  
  //mtxSpike mtxA = mtxSpike(f1.c_str(), 8);
  //mtxDense b    = mtxDense(f2.c_str());

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
  MPI_Get_processor_name(procname, &proc_name_len);

  printf("Num of task:%d, My rank:%d, My name:%s\n", size, rank, procname);
  
  blockLU(rank, f1);
  
  MPI_Finalize();

  return 0;
}



void blockLU(int rank, string &s){
  double *arr_orig=NULL;
  double *arr =NULL;
  double *L   =NULL;
  double *&U   =arr;
  double *G   =NULL;
  
  MKL_INT *ipiv =NULL;

  MKL_INT m, mm;

  printf("welcom%d\n",rank);
  if(rank == ROOT_ID) {
    mtxBLU A=mtxBLU(s.c_str(),NPROC);
    m=A.m;
    arr_orig = A.arr;
    ////printf("mst1\n");
    ////for(int k=0;k<NPROC*NPROC; k++){
    ////  int mm=m/8;
    ////  printf("BLOCK %d ===\n",k );
    ////  for(int i=0;i<mm*mm;i++){
    ////    printf("%.2f ", arr_orig[k*mm*mm+i]);
    ////  }
    ////  printf("\n");
    ////}
  }

  ////MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(&m, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
  mm   = m/NPROC;
  if(rank<=1)
    printf("%d: m%d mm%d\n",rank, m,mm);
  
  
  arr  = new double[NPROC*mm*mm];
  L    = new double[NPROC*mm*mm];
  G    = new double[      mm*mm];
  ipiv = new MKL_INT[mm        ];
  std::fill(L,   L+NPROC*mm*mm, 0);

  for(int i=0; i<mm; i++)
    L[rank*mm*mm+i+i*mm]=1;  //make Lii = eyes(mm)

  if(rank==0)
    printf("mst2\n");
  
  MPI_Scatter(arr_orig, NPROC*mm*mm, MPI_DOUBLE,
              arr     , NPROC*mm*mm, MPI_DOUBLE,
              ROOT_ID , MPI_COMM_WORLD);

  for(MKL_INT p=0; p<NPROC; p++){
    if(rank == p){
      //inv(Aii)
      double *Up = U + rank*mm*mm;
      LAPACKE_dgetrf (LAPACK_COL_MAJOR, mm, mm, Up, mm, ipiv );
      LAPACKE_dgetri (LAPACK_COL_MAJOR, mm,     Up, mm, ipiv );
    }//endof if
    for(int i=p+1; i<NPROC; i++){
      if(rank==p){
        //Gi stored into Li
        cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, mm, mm, mm, 1, U+i*mm*mm, mm, U+rank*mm*mm, mm, 0, L+i*mm*mm, mm);
        MPI_Bcast(L+i*mm*mm, mm*mm, MPI_DOUBLE, p, MPI_COMM_WORLD);
        std::fill(U+i*mm*mm, U+i*mm*mm+mm,0);
      }
      else if(rank > p){
        MPI_Bcast(G, mm*mm, MPI_DOUBLE, p, MPI_COMM_WORLD);
        MKL_INT j=i+rank-p;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mm, mm, mm, -1, G, mm, U+i*mm*mm, mm, 1, U+p*mm*mm, mm);//Aij-=G*Ajp
      }
      else{
        MPI_Bcast(G, mm*mm, MPI_DOUBLE, p, MPI_COMM_WORLD);
      }
    }//endof for i

  }//endof for p



  delete []arr; //
  delete []L; 
  delete []G;
  delete []ipiv;

}//endof function





/*************************************************************************
    > File Name: lab4.cpp
    > Author: xc
    > Descriptions: 
    > Created Time: Tue May 10 17:10:37 2016
 ************************************************************************/

#include"lab3.h"
using namespace std;


bool check_type_BLU(FILE* fp, int &M, int &N){  
  MM_typecode matcode;
  if( mm_read_banner(fp, &matcode)!=0 || 
      mm_is_matrix(matcode)!=1        ||
      mm_is_dense(matcode)!=1         ||
      mm_is_array(matcode)!=1         ||
      mm_is_real(matcode)!=1          ||
      mm_is_general(matcode)!=1       ||
      mm_read_mtx_array_size(fp, &M, &N)!=0 ) 
    return false;
  return true;
}

bool check_type_spike(FILE* fp, int &M, int &N, int &NNZ){
  MM_typecode matcode;
  if( mm_read_banner(fp, &matcode)!=0 || 
      mm_is_matrix(matcode)!=1        ||
      mm_is_sparse(matcode)!=1        ||
      mm_is_coordinate(matcode)!=1    ||
      mm_is_real(matcode)!=1          ||
      mm_is_general(matcode)!=1       ||
      mm_read_mtx_crd_size(fp, &M, &N, &NNZ)!=0 ) 
      return false;
  return true;
}


void show_mm_type(FILE* fp){//for debug
  MM_typecode matcode;
  if(mm_read_banner(fp, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    return;
  }
  if( mm_is_matrix(matcode) ) printf("is matrix\n");
  if( mm_is_sparse(matcode) ) printf("is sparse\n");
  if( mm_is_coordinate(matcode) ) printf("is coordinate\n");
  if( mm_is_dense(matcode) ) printf("is dense\n");
  if( mm_is_array(matcode) ) printf("is array\n");
  if( mm_is_complex(matcode) ) printf("is complex\n");
  if( mm_is_real(matcode) ) printf("is real\n");
  if( mm_is_pattern(matcode) ) printf("is pattern\n");
  if( mm_is_integer(matcode) ) printf("is integer\n");
  if( mm_is_symmetric(matcode) ) printf("is symm\n");
  if( mm_is_general(matcode) ) printf("is general\n");
  if( mm_is_skew(matcode) ) printf("is skew\n");
  if( mm_is_hermitian(matcode) ) printf("is hermitian\n");
}

mtxBLU::mtxBLU(const char* const s): arr(NULL){
  FILE *fp;
  if( (fp=fopen(s, "r"))==NULL  ||
      !check_type_BLU(fp, m, n) ||
      n!=1                      ||
      !getmem() ){
    cerr<<"cann't read mtx: "<<s<<endl; 
    exit(1); 
  }

  for(int i=0; i<m*1; i++){
    fscanf(fp, "%lf\n", arr+i);
  }
  fclose(fp);     
}



mtxBLU::mtxBLU(const char* const s, int _p):p(_p), arr(NULL){
  FILE *fp;
  if( (fp=fopen(s, "r"))==NULL  ||
      !check_type_BLU(fp, m, n) ||
      n!=m                      ||
      !getmem() ){
    cerr<<"cann't read mtx: "<<s<<endl; 
    exit(1); 
  }

  mm=m/p;


  int r,c,I,i,J,j;
  for(int cnt=0; cnt<m*n;cnt++){
    r=(cnt%m);
    c=int(cnt/m);

    I = int(r/mm);
    i = r%mm;

    J = int(c/mm);
    j = c%mm;

    fscanf(fp, "%lf\n", arr+ (I+J*p)*mm*mm + i+j*mm);
  }

  fclose(fp);     
}


void mtxBLU::dump_to_file(const char* const s){
  FILE *fp;
  if(arr==NULL || (fp=fopen(s, "w"))==NULL){
    cerr<<"err write mtx into file: "<<s<<endl;
    exit(1);
  }

  MM_typecode matcode;                        
  mm_initialize_typecode(&matcode);
  mm_set_matrix (&matcode);
  mm_set_dense  (&matcode);
  //mm_set_array  (&matcode);
  mm_set_real   (&matcode);
  mm_set_general(&matcode);

  mm_write_banner(fp, matcode); 
  mm_write_mtx_array_size(fp, m, n);
  for(int j=0;j<n; j++)
    for(int i=0;i<m; i++)
      fprintf(fp, "%d %d %f\n",i,j, arr[i +j*m]);
  fclose(fp);
}










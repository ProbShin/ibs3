#ifndef _LAB3_HPP
#define _LAB3_HPP

/*************************************************************************
    > File Name: lab4.hpp
    > Author: xc
    > Descriptions: 
    > Created Time: Tue May 10 17:11:03 2016
 ************************************************************************/


#include<iostream>
#include"mmio.h"

#include<cstdio>
#include<cstdlib>


#define BANDSTORAGE  1


using namespace std;

bool check_type_dense(FILE* fp, int &M, int &N);
bool check_type_spike(FILE* fp, int &M, int &N, int &NNZ);
void show_mm_type(FILE* fp);  

class mtxBLU{
  public:
    mtxBLU(const char* const s, int _p);
    ~mtxBLU(){if(arr!=NULL) delete[] arr;}
  public:
    double *arr;
    int m,n;
    int p;
    int mm;
  public:
    double* pAij(int i,int j) {return arr+i+j*m; };
    void dump_to_file(const char* const s);
  private:
    bool getmem(){ if(arr!=NULL) return false;
      return (arr=new double[m*n])!=NULL;
    }

    
};



#endif



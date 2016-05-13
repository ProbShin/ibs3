MPICC = mpiicc
COMPILE = mpiicc #mpiicc #mpicxx

MKL_MIC_ENABLE=1

MKL =    ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_ilp64.a -Wl,--end-group -liomp5 -lpthread -lm

#INCLUDES = -I. -I${MLKROOT}/include

FLAGS= -DMKL_ILP64 -qopenmp -std=c++11 -I${MKLROOT}/include 


a.out: lab3.o mmio.o main.o
	$(COMPILE) $(FLAGS) lab3.o mmio.o part_a.o -o $@ $(MKL) 

lab4.o:
	$(COMPILE) $(FLAGS) -c lab3.cpp  

mmio.o:
	$(COMPILE) $(FLAGS) -c mmio.cpp  -Wno-write-strings

part_a.o:
	$(COMPILE) $(FLAGS) -c part_a.cpp 

gmres_test: gmres_test.o mmio.o main.o lab3.o 
		$(MPICC) $(FLAGS) gmress.o mmio.o part_a.o lab3.o  -o  gmres_test $(MKL)
			
gmres.o:
	$(COMPILE) $(FLAGS) -c gmres_test.c  

run:
	mpirun -np 8 -f ./myhost -perhost 1 -genv I_MPI_DEVICE=ssm  -genv OMP_NUM_THREADS 8 ./a.out ./tinyA_dd.mtx ./tinyb_dd.mtx ./mysolution.mtx




clean:
		rm -f a.out
		rm -f *.o

# Makefile — Equação do calor 2D FTCS com solução exata

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -g -fno-omit-frame-pointer -fno-optimize-sibling-calls
OMPFLAGS ?= -fopenmp
MPIFLAGS ?= -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1

COMMON = heat2d_explicit_common.hpp

OPENMP_TARGETS = \
  heat2d_explicit_omp_naive \
  heat2d_explicit_omp_naive_nofs \
  heat2d_explicit_omp_busywait_nobarrier \
  heat2d_explicit_omp_busywait_nobarrier_nofs \
  heat2d_explicit_omp_sem_nobarrier \
  heat2d_explicit_omp_sem_nobarrier_nofs \
  heat2d_explicit_omp_mpilike

MPI_TARGETS = heat2d_explicit_mpi_naive_1d

.PHONY: all openmp mpi clean

all: openmp mpi

openmp: $(OPENMP_TARGETS)

mpi: $(MPI_TARGETS)

heat2d_explicit_omp_naive: heat2d_explicit_omp_naive.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_omp_naive_nofs: heat2d_explicit_omp_naive_nofs.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_omp_busywait_nobarrier: heat2d_explicit_omp_busywait_nobarrier.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_omp_busywait_nobarrier_nofs: heat2d_explicit_omp_busywait_nobarrier_nofs.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_omp_sem_nobarrier: heat2d_explicit_omp_sem_nobarrier.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_omp_sem_nobarrier_nofs: heat2d_explicit_omp_sem_nobarrier_nofs.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_omp_mpilike: heat2d_explicit_omp_mpilike.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

heat2d_explicit_mpi_naive_1d: heat2d_explicit_mpi_naive_1d.cpp
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) $< -o $@

clean:
	rm -f $(OPENMP_TARGETS) $(MPI_TARGETS) output.txt *.o

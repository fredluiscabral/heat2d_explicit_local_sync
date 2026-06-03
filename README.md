# heat2d_explicit_local_sync — versão com solução exata

Equação do calor 2D resolvida pelo método totalmente explícito FTCS, com variantes OpenMP, MPI puro e OpenMP MPI-like.

## Equação

```text
u_t = alpha (u_xx + u_yy),  (x,y) em [0,1] x [0,1]
```

com contorno de Dirichlet homogêneo.

## Solução exata usada para verificação

A condição inicial foi alterada para permitir comparação analítica:

```text
u(x,y,0) = sin(pi x) sin(pi y)
```

A solução exata é:

```text
u(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 alpha t)
```

Todas as variantes imprimem:

```text
L1_error
L2_error
Linf_error
Tempo
```

## Esquema FTCS

```text
U^{n+1}_{i,j} =
    U^n_{i,j}
  + lambda (U^n_{i+1,j} + U^n_{i-1,j} + U^n_{i,j+1} + U^n_{i,j-1} - 4U^n_{i,j})
```

com

```text
dt = 0.90*h*h/(4*alpha)
lambda = alpha*dt/(h*h)
```

## Variantes

- `heat2d_explicit_omp_naive.cpp`
- `heat2d_explicit_omp_naive_nofs.cpp`
- `heat2d_explicit_omp_busywait_nobarrier.cpp`
- `heat2d_explicit_omp_busywait_nobarrier_nofs.cpp`
- `heat2d_explicit_omp_sem_nobarrier.cpp`
- `heat2d_explicit_omp_sem_nobarrier_nofs.cpp`
- `heat2d_explicit_omp_mpilike.cpp`
- `heat2d_explicit_mpi_naive_1d.cpp`

## Compilação no SDumont

```bash
module load amd-compilers/5.0.0
module load amd-libraries/5.0.0
module load openmpi/amd/5.0

make clean
make all
```

## Execução

```bash
cd experimentos
sbatch submeter.sh
```

Depois:

```bash
./consolidar.sh results_<JOBID>
cat results_<JOBID>/summary_cases.tsv
```

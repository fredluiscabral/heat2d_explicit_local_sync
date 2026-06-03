#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --ntasks=192
#SBATCH --exclusive
#SBATCH -p lncc-cpu_amd
#SBATCH -J heat2d-exact
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=3-00:00:00

set -euo pipefail
export LC_ALL=C

# ==============================================================================
# CONFIGURAÇÃO DO TESTE — EDITE AQUI
# ==============================================================================

# A primeira repetição é warmup. REPS=11 => 1 warmup + 10 válidas.
REPS=${REPS:-11}

WORKERS_LIST=(1 4 16 64 128 192)

VARIANTS_LIST=(
    mpi_puro
    omp_naive
    omp_naive_nofs
    omp_busywait
    omp_busywait_nofs
    omp_semaforos
    omp_semaforos_nofs
    omp_mpilike
)

OMP_PROC_BIND_VALUE="${OMP_PROC_BIND_VALUE:-close}"
USE_NUMACTL_INTERLEAVE="${USE_NUMACTL_INTERLEAVE:-0}"

SRUN_MAX_TRIES="${SRUN_MAX_TRIES:-3}"
SRUN_RETRY_SLEEP="${SRUN_RETRY_SLEEP:-30}"

# ==============================================================================
# FIM DA CONFIGURAÇÃO
# ==============================================================================

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NA}"
if command -v nodeset >/dev/null 2>&1 && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    nodeset -e "$SLURM_JOB_NODELIST"
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

PARAM_FILE="$SCRIPT_DIR/param.txt"
if [[ ! -f "$PARAM_FILE" ]]; then
    echo "ERRO: arquivo obrigatório não encontrado: $PARAM_FILE" >&2
    exit 1
fi

echo
echo "=== Diretórios ==="
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "PARAM_FILE=$PARAM_FILE"

echo
echo "=== Configuração do teste ==="
echo "REPS=$REPS"
echo "WORKERS_LIST=${WORKERS_LIST[*]}"
echo "VARIANTS_LIST=${VARIANTS_LIST[*]}"
echo "OMP_PROC_BIND_VALUE=$OMP_PROC_BIND_VALUE"
echo "USE_NUMACTL_INTERLEAVE=$USE_NUMACTL_INTERLEAVE"
echo "SRUN_MAX_TRIES=$SRUN_MAX_TRIES"
echo "SRUN_RETRY_SLEEP=$SRUN_RETRY_SLEEP"

echo
echo "=== param.txt ==="
cat "$PARAM_FILE"

module load amd-compilers/5.0.0
module load amd-libraries/5.0.0
module load openmpi/amd/5.0

export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND="$OMP_PROC_BIND_VALUE"

variant_exec() {
    local v="$1"

    case "$v" in
        mpi_puro)
            echo "$PROJECT_DIR/heat2d_explicit_mpi_naive_1d"
            ;;
        omp_naive)
            echo "$PROJECT_DIR/heat2d_explicit_omp_naive"
            ;;
        omp_naive_nofs)
            echo "$PROJECT_DIR/heat2d_explicit_omp_naive_nofs"
            ;;
        omp_busywait)
            echo "$PROJECT_DIR/heat2d_explicit_omp_busywait_nobarrier"
            ;;
        omp_busywait_nofs)
            echo "$PROJECT_DIR/heat2d_explicit_omp_busywait_nobarrier_nofs"
            ;;
        omp_semaforos)
            echo "$PROJECT_DIR/heat2d_explicit_omp_sem_nobarrier"
            ;;
        omp_semaforos_nofs)
            echo "$PROJECT_DIR/heat2d_explicit_omp_sem_nobarrier_nofs"
            ;;
        omp_mpilike)
            echo "$PROJECT_DIR/heat2d_explicit_omp_mpilike"
            ;;
        *)
            echo "ERRO: variante desconhecida: $v" >&2
            return 1
            ;;
    esac
}

variant_kind() {
    local v="$1"
    case "$v" in
        mpi_puro) echo "mpi" ;;
        *)        echo "omp" ;;
    esac
}

echo
echo "=== Checando executáveis selecionados ==="
for variant in "${VARIANTS_LIST[@]}"; do
    exe="$(variant_exec "$variant")"

    if [[ ! -x "$exe" ]]; then
        echo "ERRO: executável não encontrado ou sem permissão de execução:" >&2
        echo "  variante: $variant" >&2
        echo "  executável: $exe" >&2
        echo >&2
        echo "Dica:" >&2
        echo "  cd $PROJECT_DIR" >&2
        echo "  make all" >&2
        exit 1
    fi

    echo "$variant -> $exe"
done

OUTROOT="$SCRIPT_DIR/results_${SLURM_JOB_ID}"
mkdir -p "$OUTROOT"

MANIFEST="$OUTROOT/manifest.tsv"
printf "label\tvariant\tkind\tworkers\tranks\tthreads\trep\twarmup\trundir\tstdout\tstderr\tparam\tattempts\tomp_proc_bind\tnumactl_interleave\n" > "$MANIFEST"

run_srun_with_retry() {
    local rdir="$1"
    local ranks="$2"
    local threads="$3"
    local exe="$4"
    local stdout_file="$5"
    local stderr_file="$6"
    local kind="$7"

    local try=1
    local status=0

    while true; do
        echo "Tentativa srun $try/$SRUN_MAX_TRIES"

        (
            cd "$rdir"
            export OMP_NUM_THREADS="$threads"

            set +e

            if [[ "$kind" == "omp" && "$USE_NUMACTL_INTERLEAVE" == "1" ]]; then
                srun --exclusive --exact -N 1 -n "$ranks" -c "$threads" \
                     --hint=nomultithread \
                     --distribution=block:block \
                     --cpu-bind=cores \
                     numactl --interleave=all "$exe" \
                     > "$stdout_file" 2> "$stderr_file"
            else
                srun --exclusive --exact -N 1 -n "$ranks" -c "$threads" \
                     --hint=nomultithread \
                     --distribution=block:block \
                     --cpu-bind=cores \
                     "$exe" \
                     > "$stdout_file" 2> "$stderr_file"
            fi

            exit $?
        )

        status=$?

        if [[ "$status" -eq 0 ]]; then
            ATTEMPTS_USED="$try"
            return 0
        fi

        echo "Aviso: srun falhou com status $status na tentativa $try/$SRUN_MAX_TRIES." >&2
        echo "stderr da tentativa:" >&2
        if [[ -f "$stderr_file" ]]; then
            cat "$stderr_file" >&2 || true
        fi

        if [[ "$try" -ge "$SRUN_MAX_TRIES" ]]; then
            echo "ERRO: srun falhou após $SRUN_MAX_TRIES tentativas." >&2
            ATTEMPTS_USED="$try"
            return "$status"
        fi

        echo "Aguardando ${SRUN_RETRY_SLEEP}s antes de tentar novamente..." >&2
        sleep "$SRUN_RETRY_SLEEP"
        try=$((try + 1))
    done
}

run_case() {
    local variant="$1"
    local workers="$2"

    local kind
    local exe
    local ranks
    local threads

    kind="$(variant_kind "$variant")"
    exe="$(variant_exec "$variant")"

    if [[ "$kind" == "mpi" ]]; then
        ranks="$workers"
        threads=1
    else
        ranks=1
        threads="$workers"
    fi

    local label
    label="$(printf "%s_R%03d_T%03d_W%03d" "$variant" "$ranks" "$threads" "$workers")"

    local cdir
    cdir="$OUTROOT/$label"
    mkdir -p "$cdir"

    echo
    echo "============================================================"
    echo "Caso: $label"
    echo "Variante: $variant"
    echo "Tipo: $kind"
    echo "Executável: $exe"
    echo "workers=$workers | ranks=$ranks | threads=$threads"
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
    echo "USE_NUMACTL_INTERLEAVE=$USE_NUMACTL_INTERLEAVE"
    echo "============================================================"

    for rep in $(seq 1 "$REPS"); do
        local warmup_flag
        local tag
        local rdir
        local stdout_file
        local stderr_file

        if [[ "$rep" -eq 1 ]]; then
            warmup_flag=1
            tag="warmup"
        else
            warmup_flag=0
            tag="$(printf "rep%02d" "$rep")"
        fi

        rdir="$cdir/$tag"
        mkdir -p "$rdir"

        cp "$PARAM_FILE" "$rdir/param.txt"

        stdout_file="$rdir/stdout.txt"
        stderr_file="$rdir/stderr.txt"

        echo
        echo "--- $label | rodada $rep/$REPS | warmup=$warmup_flag ---"
        date

        ATTEMPTS_USED=0
        run_srun_with_retry "$rdir" "$ranks" "$threads" "$exe" "$stdout_file" "$stderr_file" "$kind"

        printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$label" "$variant" "$kind" "$workers" "$ranks" "$threads" "$rep" "$warmup_flag" \
            "$rdir" "$stdout_file" "$stderr_file" "$rdir/param.txt" "$ATTEMPTS_USED" \
            "$OMP_PROC_BIND_VALUE" "$USE_NUMACTL_INTERLEAVE" >> "$MANIFEST"
    done
}

for workers in "${WORKERS_LIST[@]}"; do
    for variant in "${VARIANTS_LIST[@]}"; do
        run_case "$variant" "$workers"
    done
done

echo
echo "Job finalizado."
echo "Resultados em: $OUTROOT"
echo "Manifesto: $MANIFEST"
echo
echo "ATENÇÃO:"
echo "  Cada caso tem $REPS execuções."
echo "  Use apenas as repetições 2..$REPS na consolidação."
echo "  A primeira execução de cada caso é warmup."

#!/bin/bash
set -euo pipefail
export LC_ALL=C

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Uso: $0 <DIR_RESULTS> [ARQ_SAIDA.tsv]" >&2
    echo "Exemplo:" >&2
    echo "  $0 results_438460" >&2
    exit 1
fi

RESULTS_DIR="$1"
OUT_TSV="${2:-$RESULTS_DIR/summary_cases.tsv}"
EXPECTED_VALID_REPS="${EXPECTED_VALID_REPS:-10}"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Erro: diretório não encontrado: $RESULTS_DIR" >&2
    exit 1
fi

python3 - "$RESULTS_DIR" "$OUT_TSV" "$EXPECTED_VALID_REPS" <<'PY'
import sys
import re
import csv
import statistics
from pathlib import Path

results_dir = Path(sys.argv[1])
out_tsv = Path(sys.argv[2])
expected_valid_reps = int(sys.argv[3])

num_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def extract_metric(text, key):
    m = re.findall(rf"^\s*{re.escape(key)}\s*:\s*({num_re})", text, flags=re.MULTILINE)
    return float(m[-1]) if m else None

def extract_string_metric(text, key):
    m = re.findall(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return m[-1].strip() if m else "NA"

def parse_label(label):
    m = re.match(r"^(.+)_R(\d+)_T(\d+)_W(\d+)$", label)
    if not m:
        raise ValueError(f"nome de caso fora do padrão esperado: {label}")
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))

def collect_rep_files(case_dir):
    files = sorted(case_dir.glob("rep*/stdout.txt"))
    if not files:
        files = sorted(case_dir.glob("stdout_rep*.txt"))
    return files

def stats(values):
    if not values:
        return {"mean": "NA", "median": "NA", "min": "NA", "max": "NA", "raw": ""}
    vals = [float(v) for v in values]
    vals_sorted = sorted(vals)
    return {
        "mean": f"{statistics.fmean(vals):.10f}",
        "median": f"{statistics.median(vals_sorted):.10f}",
        "min": f"{vals_sorted[0]:.10f}",
        "max": f"{vals_sorted[-1]:.10f}",
        "raw": ",".join(f"{v:.17g}" for v in vals),
    }

def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.17g}"
    return str(x)

rows = []

for case_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
    label = case_dir.name

    if not re.match(r"^.+_R\d+_T\d+_W\d+$", label):
        continue

    variant, ranks, threads, workers = parse_label(label)
    rep_files = collect_rep_files(case_dir)

    if not rep_files:
        print(f"Aviso: nenhuma repetição válida em {case_dir}", file=sys.stderr)
        continue

    texts = [(f, f.read_text(encoding="utf-8", errors="replace")) for f in rep_files]
    first_text = texts[0][1]

    tempos, l1s, l2s, linfs = [], [], [], []

    for f, txt in texts:
        tempo = extract_metric(txt, "Tempo")
        if tempo is None:
            print(f"Erro: não consegui extrair Tempo de {f}", file=sys.stderr)
            sys.exit(1)
        tempos.append(tempo)

        l1 = extract_metric(txt, "L1_error")
        l2 = extract_metric(txt, "L2_error")
        linf = extract_metric(txt, "Linf_error")

        if l1 is not None: l1s.append(l1)
        if l2 is not None: l2s.append(l2)
        if linf is not None: linfs.append(linf)

    nvalid = len(tempos)
    if nvalid != expected_valid_reps:
        print(f"Aviso: caso {label} tem {nvalid} repetições válidas; esperado {expected_valid_reps}", file=sys.stderr)

    tempo_stats = stats(tempos)
    l1_stats = stats(l1s)
    l2_stats = stats(l2s)
    linf_stats = stats(linfs)

    rows.append({
        "label": label,
        "variant": variant,
        "variant_reported": extract_string_metric(first_text, "Variant"),
        "workers": workers,
        "ranks": ranks,
        "threads": threads,
        "nvalid": nvalid,
        "N": fmt(extract_metric(first_text, "N")),
        "alpha": fmt(extract_metric(first_text, "alpha")),
        "steps": fmt(extract_metric(first_text, "T")),
        "TILE": fmt(extract_metric(first_text, "TILE")),
        "dt": fmt(extract_metric(first_text, "dt")),
        "lambda": fmt(extract_metric(first_text, "lambda")),
        "final_time": fmt(extract_metric(first_text, "final_time")),
        "mean_s": tempo_stats["mean"],
        "median_s": tempo_stats["median"],
        "min_s": tempo_stats["min"],
        "max_s": tempo_stats["max"],
        "mean_l1": l1_stats["mean"],
        "median_l1": l1_stats["median"],
        "min_l1": l1_stats["min"],
        "max_l1": l1_stats["max"],
        "mean_l2": l2_stats["mean"],
        "median_l2": l2_stats["median"],
        "min_l2": l2_stats["min"],
        "max_l2": l2_stats["max"],
        "mean_linf": linf_stats["mean"],
        "median_linf": linf_stats["median"],
        "min_linf": linf_stats["min"],
        "max_linf": linf_stats["max"],
        "raw_s": tempo_stats["raw"],
        "raw_l1": l1_stats["raw"],
        "raw_l2": l2_stats["raw"],
        "raw_linf": linf_stats["raw"],
    })

columns = [
    "label", "variant", "variant_reported", "workers", "ranks", "threads", "nvalid",
    "N", "alpha", "steps", "TILE", "dt", "lambda", "final_time",
    "mean_s", "median_s", "min_s", "max_s",
    "mean_l1", "median_l1", "min_l1", "max_l1",
    "mean_l2", "median_l2", "min_l2", "max_l2",
    "mean_linf", "median_linf", "min_linf", "max_linf",
    "raw_s", "raw_l1", "raw_l2", "raw_linf",
]

rows.sort(key=lambda r: (int(r["workers"]), str(r["variant"])))

out_tsv.parent.mkdir(parents=True, exist_ok=True)
with out_tsv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

print(f"Resumo gravado em: {out_tsv}")

print()
print("Ranking por tempo mediano em cada número de workers:")
by_workers = {}
for r in rows:
    try:
        med = float(r["median_s"])
    except Exception:
        continue
    by_workers.setdefault(int(r["workers"]), []).append((med, r["variant"], r["label"]))

for w in sorted(by_workers):
    best = sorted(by_workers[w], key=lambda x: x[0])[0]
    print(f"workers={w}\tbest={best[1]}\tmedian_s={best[0]:.10f}\tlabel={best[2]}")

print()
print("Ranking completo por workers:")
for w in sorted(by_workers):
    print(f"\nworkers={w}")
    for pos, (med, variant, label) in enumerate(sorted(by_workers[w], key=lambda x: x[0]), start=1):
        print(f"{pos}\t{variant}\tmedian_s={med:.10f}\t{label}")
PY

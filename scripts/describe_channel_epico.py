"""Descriptive characterisation of the epiCo channel product (intermediate unit before F2).

Reads ONLY data/processed/epidemiology/pilot_weekly_channel.parquet (after checking its md5 against
datos.version_dvc_canal) and writes figures, CSV tables and a README. It computes no alert metric
against the onsets (that is F4, contrato §10.15) and never reads the legacy labels parquet.

Definitions (same as tests/acceptance/test_f1_channel_product.py):
- run: maximal block of consecutive eligible weeks with target_u1e = 1;
- onset (Umaña et al., U1e): first week of a run of >= 3 weeks preceded by >= 2 weeks with 0.

Usage:
    .venv/Scripts/python.exe scripts/describe_channel_epico.py \
        --out-dir reports/epidemiology/evidence/channel_20261111
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNT = "confirmed_records_screening"
ZONES = ("seguridad", "alerta", "epidemia")
# Display order follows AGENTS.md (pilot list).
MUNICIPALITIES = [
    ("54001", "Cúcuta"), ("08001", "Barranquilla"), ("41001", "Neiva"),
    ("50001", "Villavicencio"), ("76001", "Cali"), ("08758", "Soledad"),
]
# Expected figures: especificación §6.F1 / contrato §2.2 (checked, never used to alter data).
EXPECTED_TOTALS = {"eligible": 4380, "positives": 945, "runs": 175, "onsets": 31}
EXPECTED_ONSETS = {"54001": 3, "08001": 5, "41001": 6, "50001": 7, "76001": 3, "08758": 7}
EXPECTED_ZONE_SHARES = (0.379, 0.405, 0.216)
DEV_RANGE = (2019, 2023)    # [2019-01-01, 2023-01-01): development, reference only
TEST_RANGE = (2023, 2026)   # [2023-01-01, 2026-01-01): test, reference only

# Zone fills (light, warm progression), lines and markers.
C_ZONE = {"seguridad": "#d9ecd9", "alerta": "#fbe3b0", "epidemia": "#f3bcb6"}
C_CENTRAL, C_UPPER, C_CASES, C_ONSET = "#2f6b3a", "#a3261b", "#111111", "#1f3a93"
C_DEV, C_TEST = "#4a6fa5", "#7a4a9a"


# --------------------------------------------------------------------------- inputs
def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_channel(path: Path, expected_md5: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Channel product not found: {path}")
    actual = md5_of(path)
    if actual != expected_md5:
        raise ValueError(f"md5 mismatch for {path}: disk {actual} != expected {expected_md5}")
    frame = pd.read_parquet(path)
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    eligible = frame[frame["channel_eligible"]].sort_values(["municipality_code", "week_start"])
    return eligible.reset_index(drop=True)


# --------------------------------------------------------------------------- events
def runs_of(values: np.ndarray) -> list[tuple[int, int]]:
    """(start index, length) of each maximal block of ones."""
    runs, i = [], 0
    while i < len(values):
        if values[i] == 1:
            j = i
            while j < len(values) and values[j] == 1:
                j += 1
            runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def add_events(eligible: pd.DataFrame) -> pd.DataFrame:
    """Add boolean columns run_start and onset (per municipality, over eligible weeks)."""
    eligible = eligible.copy()
    eligible["run_start"] = False
    eligible["onset"] = False
    for _, index in eligible.groupby("municipality_code").groups.items():
        values = eligible.loc[index, "target_u1e"].astype(int).to_numpy()
        for start, length in runs_of(values):
            eligible.loc[index[start], "run_start"] = True
            if length >= 3 and start >= 2 and values[start - 1] == 0 and values[start - 2] == 0:
                eligible.loc[index[start], "onset"] = True
    return eligible


# --------------------------------------------------------------------------- tables
def year_table(eligible: pd.DataFrame) -> pd.DataFrame:
    names = dict(MUNICIPALITIES)
    eligible = eligible.assign(year=eligible["week_start"].dt.year)
    rows = []
    for code, label in names.items():
        group_m = eligible[eligible["municipality_code"] == code]
        parts = [(str(year), g) for year, g in group_m.groupby("year")] + [("total", group_m)]
        for year, g in parts:
            n = len(g)
            rows.append({
                "municipality_code": code, "municipality": label, "year": year,
                "eligible_weeks": n, "positive_weeks": int(g["target_u1e"].sum()),
                "prevalence": round(g["target_u1e"].mean(), 4),
                **{f"share_{z}": round((g["channel_zone"] == z).mean(), 4) for z in ZONES},
                "runs": int(g["run_start"].sum()), "onsets": int(g["onset"].sum()),
            })
    return pd.DataFrame(rows)


def width_table(eligible: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, label in MUNICIPALITIES:
        g = eligible[eligible["municipality_code"] == code]
        ratio = g["epico_upper"] / g["epico_central"]
        rows.append({
            "municipality_code": code, "municipality": label, "eligible_weeks": len(g),
            "ratio_p10": ratio.quantile(0.10), "ratio_median": ratio.median(),
            "ratio_p90": ratio.quantile(0.90),
            "weeks_central_zero": int((g["epico_central"] == 0).sum()),
            "weeks_central_below_1": int((g["epico_central"] < 1).sum()),
            "central_min": g["epico_central"].min(),
        })
    return pd.DataFrame(rows).round(4)


def boundary_jumps(eligible: pd.DataFrame) -> pd.DataFrame:
    """Median |Δ log U1e| between consecutive weeks: across a year change vs inside a year."""
    rows = []
    for code, label in MUNICIPALITIES:
        g = eligible[eligible["municipality_code"] == code]
        delta = np.abs(np.diff(np.log1p(g["epico_upper"].to_numpy())))
        year = g["week_start"].dt.year.to_numpy()
        crosses = year[1:] != year[:-1]
        rows.append({"municipality": label, "median_abs_dlog_year_change": np.median(delta[crosses]),
                     "median_abs_dlog_within_year": np.median(delta[~crosses])})
    return pd.DataFrame(rows).round(3)


def check_reproduction(eligible: pd.DataFrame, years: pd.DataFrame) -> None:
    """Fail loudly if the descriptive totals do not reproduce §6.F1 (no adjustment)."""
    totals = years[years["year"] == "total"].set_index("municipality_code")
    observed = {"eligible": int(totals["eligible_weeks"].sum()), "positives": int(totals["positive_weeks"].sum()),
                "runs": int(totals["runs"].sum()), "onsets": int(totals["onsets"].sum())}
    if observed != EXPECTED_TOTALS:
        raise AssertionError(f"Totals do not reproduce §6.F1: {observed} != {EXPECTED_TOTALS}")
    by_code = totals["onsets"].astype(int).to_dict()
    if by_code != EXPECTED_ONSETS:
        raise AssertionError(f"Onsets by municipality {by_code} != {EXPECTED_ONSETS}")
    shares = tuple(round(float((eligible["channel_zone"] == z).mean()), 3) for z in ZONES)
    if shares != EXPECTED_ZONE_SHARES:
        raise AssertionError(f"Zone shares {shares} != {EXPECTED_ZONE_SHARES}")


# --------------------------------------------------------------------------- figures
def draw_channel(ax, g: pd.DataFrame, *, log: bool, title: str | None, show_onsets: bool = True) -> None:
    """Zones, central, U1e, cases and onsets for one municipality. Full range, no clipping."""
    transform = np.log1p if log else (lambda v: v)
    x = g["week_start"].to_numpy()
    central = transform(g["epico_central"].to_numpy())
    upper = transform(g["epico_upper"].to_numpy())
    cases = transform(g[COUNT].to_numpy().astype(float))
    ymax = float(max(cases.max(), upper.max())) * 1.05
    # One value per week: step drawing, no interpolation between weeks.
    ax.fill_between(x, 0, central, step="post", color=C_ZONE["seguridad"], lw=0)
    ax.fill_between(x, central, upper, step="post", color=C_ZONE["alerta"], lw=0)
    ax.fill_between(x, upper, ymax, step="post", color=C_ZONE["epidemia"], lw=0)
    ax.step(x, central, where="post", color=C_CENTRAL, lw=0.9)
    ax.step(x, upper, where="post", color=C_UPPER, lw=0.9)
    ax.plot(x, cases, color=C_CASES, lw=0.7)
    # Weeks with central = 0 (none in the current product): the safety band has zero height.
    zero = x[g["epico_central"].to_numpy() == 0]
    if len(zero):
        ax.plot(zero, np.zeros(len(zero)), "|", color="#555555", ms=7, clip_on=False)
    for year in range(2013, 2026):
        ax.axvline(pd.Timestamp(year, 1, 1), color="#8a8a8a", lw=0.4, ls=":")
    if show_onsets:
        onsets = g["onset"].to_numpy()
        ax.plot(x[onsets], cases[onsets], "v", color=C_ONSET, mec="white", mew=0.6, ms=6.5, zorder=5)
    ax.set_ylim(0, ymax)
    ax.set_xlim(pd.Timestamp(2012, 1, 1), pd.Timestamp(2026, 1, 1))
    ax.margins(x=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=14)
    # Reference strip (development / test) above the axis; it does not tint the zones.
    strip = ax.inset_axes([0, 1.0, 1, 0.035], transform=ax.transAxes)
    strip.set_xlim(ax.get_xlim())
    strip.axvspan(pd.Timestamp(DEV_RANGE[0], 1, 1), pd.Timestamp(DEV_RANGE[1], 1, 1), color=C_DEV, alpha=0.55, lw=0)
    strip.axvspan(pd.Timestamp(TEST_RANGE[0], 1, 1), pd.Timestamp(TEST_RANGE[1], 1, 1), color=C_TEST, alpha=0.55, lw=0)
    strip.axis("off")


def legend_handles() -> list:
    return [
        Patch(facecolor=C_ZONE["seguridad"], label="Zona de seguridad (0 – central)"),
        Patch(facecolor=C_ZONE["alerta"], label="Zona de alerta (central – U1e)"),
        Patch(facecolor=C_ZONE["epidemia"], label="Zona de epidemia (> U1e)"),
        Line2D([], [], color=C_CENTRAL, lw=1.2, label="Línea central"),
        Line2D([], [], color=C_UPPER, lw=1.2, label="U1e"),
        Line2D([], [], color=C_CASES, lw=1, label="Casos semanales"),
        Line2D([], [], marker="v", color="none", mfc=C_ONSET, mec="white", ms=8, label="Inicio de brote (Umaña, U1e)"),
        Patch(facecolor=C_DEV, alpha=0.55, label="Desarrollo 2019–2022 (referencia)"),
        Patch(facecolor=C_TEST, alpha=0.55, label="Prueba 2023–2025 (referencia)"),
    ]


def save_all(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main_figure(eligible: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 11.5), sharex=True)
    for ax, (code, label) in zip(axes.ravel(), MUNICIPALITIES):
        g = eligible[eligible["municipality_code"] == code]
        draw_channel(ax, g, log=False, title=f"{label} ({int(g['onset'].sum())} inicios)")
        ax.set_ylabel("Casos por semana")
    fig.suptitle("Canal de epiCo (método geométrico, n = 5): semanas elegibles 2012–2025", fontsize=13, x=0.01, ha="left")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.text(0.99, 0.985, "Eje y independiente por panel, escala lineal, sin recorte (incluye U1e completo).",
             fontsize=8, color="#444444", ha="right", va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save_all(fig, out_dir / "channel_main_six_municipalities")


def detail_figures(eligible: pd.DataFrame, out_dir: Path) -> None:
    for code, label in MUNICIPALITIES:
        g = eligible[eligible["municipality_code"] == code]
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        draw_channel(top, g, log=False, title=f"{label}: escala lineal ({int(g['onset'].sum())} inicios)")
        top.set_ylabel("Casos por semana")
        draw_channel(bottom, g, log=True, title=None)
        bottom.set_ylabel("log(C + 1)")
        bottom.set_title("Escala log(C + 1)", loc="left", fontsize=10, pad=14)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save_all(fig, out_dir / f"channel_detail_{code}_{unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode().lower()}")


# --------------------------------------------------------------------------- provenance / README
def git_info() -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--", "scripts/describe_channel_epico.py"))


def write_readme(out_dir: Path, md5: str, jumps: pd.DataFrame, widths: pd.DataFrame, argv: list[str]) -> None:
    commit, dirty = git_info()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jumps_md = "\n".join(f"| {r.municipality} | {r.median_abs_dlog_year_change} | {r.median_abs_dlog_within_year} |"
                         for r in jumps.itertuples())
    command = ".venv/Scripts/python.exe scripts/describe_channel_epico.py " + " ".join(argv)
    text = f"""# Caracterización descriptiva del canal de epiCo

Unidad intermedia antes de F2: no es una fase del contrato. Solo describe el producto del canal;
no calcula ninguna métrica de la alerta frente a los inicios (eso es F4, contrato §10.15) y no lee el parquet de legado.

## Procedencia

- Producto: `data/processed/epidemiology/pilot_weekly_channel.parquet`, md5 `{md5}` (igual a `datos.version_dvc_canal`; el script falla si no coincide).
- Commit base del código: `{commit}`{' (el script tenía cambios sin commit al generar)' if dirty else ''}.
- sha256 del script `scripts/describe_channel_epico.py`: `{script_sha}`.
- Regenerar (desde la raíz del repositorio):

```bash
{command}
```

## Archivos

| Archivo | Contenido |
|---|---|
| `channel_main_six_municipalities.png` / `.svg` | Figura principal: seis paneles, semanas elegibles 2012–2025, eje x compartido, eje y independiente, lineal y sin recorte. Zonas, central, U1e, casos, inicios de brote (31) y franja de desarrollo/prueba solo como referencia. PNG a 300 dpi. |
| `channel_detail_<código>_<municipio>.png` / `.svg` | Un detalle por municipio: escala lineal y log(C + 1). |
| `channel_by_municipality_year.csv` | Por municipio y año (y fila `total`): semanas elegibles, positivas, prevalencia, proporción en cada zona, rachas (por año de inicio) e inicios de brote. Los totales reproducen §6.F1: 4.380 semanas, 945 positivas, 175 rachas, 31 inicios; zonas 0,379 / 0,405 / 0,216. |
| `channel_width_by_municipality.csv` | Anchura del canal: mediana y cuantiles 10 y 90 de U1e/central, semanas con central = 0 y con central < 1. |

## Definiciones

- Racha: bloque máximo de semanas elegibles consecutivas con `target_u1e = 1`.
- Inicio de brote (Umaña et al., U1e): primera semana de una racha de ≥ 3 semanas precedida de ≥ 2 semanas con 0. Misma definición que la prueba de aceptación de F1.
- Bandas: seguridad de 0 a la central, alerta de la central a U1e, epidemia por encima de U1e. Cada semana se dibuja con un valor (escalón), sin interpolar.
- Franja superior de cada panel: desarrollo 2019–2022 y prueba 2023–2025, solo como referencia visual.

## Lecturas del canal

- Central nunca es 0 en las semanas elegibles (mínimo {widths['central_min'].min()}); ver `weeks_central_below_1` en la tabla de anchura.
- El canal se calcula por ordinal de semana con los cinco años anteriores, por lo que U1e es irregular de una semana a otra. Mediana de |Δ log(U1e + 1)| entre semanas consecutivas:

| Municipio | Con cambio de año | Dentro del año |
|---|---|---|
{jumps_md}
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel-parquet", type=Path,
                        default=REPO_ROOT / "data/processed/epidemiology/pilot_weekly_channel.parquet")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/forecasting.yaml")
    parser.add_argument("--expected-md5", default=None,
                        help="Default: datos.version_dvc_canal from --config.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = args.expected_md5
    if expected is None:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        expected = str(config["datos"]["version_dvc_canal"])
    eligible = add_events(load_channel(args.channel_parquet, expected))
    years = year_table(eligible)
    check_reproduction(eligible, years)
    widths, jumps = width_table(eligible), boundary_jumps(eligible)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    years.to_csv(args.out_dir / "channel_by_municipality_year.csv", index=False, encoding="utf-8")
    widths.to_csv(args.out_dir / "channel_width_by_municipality.csv", index=False, encoding="utf-8")
    main_figure(eligible, args.out_dir)
    detail_figures(eligible, args.out_dir)
    write_readme(args.out_dir, expected, jumps, widths, sys.argv[1:])
    print(f"OK: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

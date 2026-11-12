"""caso_f6b.py: independent check of the selection with the feature-set step of amendment D7.2 (contract §6.6).

Written in the chat «Desarrollo guiado con Claude Code» on 2026-11-12. It imports nothing from src/. It reuses
caso_f6.py (same folder, also independent of src/) for the out-of-year scores, steps 1-4, events and the tau
rule, and adds step 0 and the ablation definition.

Usage (from the repository root):

    python -X utf8 -I docs/forecasting/verificacion/caso_f6b.py \
        --canal data/processed/epidemiology/pilot_weekly_channel.parquet \
        --completo data/processed/forecasting --sin-clima data/processed/forecasting/sin_clima \
        --json caso_f6b_resultado.json

Step 0 (contract §6.6), per family: the default set is «sin clima»; «completo» replaces it only if, with the
out-of-year common configurations of each set, its mean improvement over B2 is larger and it has a strictly
smaller log-loss in at least 17 of the 24 cells. No per-municipality exception. Then steps 1-4 and tau with
each family's chosen set (caso_f6.select and caso_f6.tau_step). Ablation: per municipality, the final family
with the other set, with its common configuration and its hyperparameter exception (4 of 4, out of year).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caso_f6 as base  # noqa: E402  (independent verification script in the same folder)

import pandas as pd  # noqa: E402

SETS = ("sin_clima", "completo")


def step0(tables: dict, family: str) -> dict:
    oy = {s: base.out_of_year(tables[s][tables[s]["familia"] == family], family) for s in SETS}
    pair = oy["sin_clima"].merge(oy["completo"], on=["municipality_code", "anio"], suffixes=("_sin", "_com"))
    if len(pair) != 24:
        base.fail(f"{family}: expected 24 cells, got {len(pair)}")
    codes = pair["municipality_code"].nunique()
    mean = {s: math.fsum(pair.groupby("municipality_code")[f"mejora_comun_{tag}"].mean()) / codes
            for s, tag in (("sin_clima", "sin"), ("completo", "com"))}
    wins = int((pair["ll_comun_com"] < pair["ll_comun_sin"]).sum())
    ties = int((pair["ll_comun_com"] == pair["ll_comun_sin"]).sum())
    threshold = base.binomial_threshold(24, base.ALPHA)
    chosen = "completo" if (mean["completo"] > mean["sin_clima"] and wins >= threshold) else "sin_clima"
    print(f"  Paso 0, {family}: mejora media fuera de año sin clima {mean['sin_clima']:+.6f}, completo "
          f"{mean['completo']:+.6f}; completo gana {wins} de 24 (empates {ties}, umbral {threshold}) -> {chosen}")
    return {"mejora_fuera_de_anio": mean, "victorias_completo": wins, "empates": ties, "umbral": threshold,
            "conjunto": chosen,
            "casillas": pair[["municipality_code", "anio", "comun_sin", "comun_com", "ll_comun_sin", "ll_comun_com"]]
            .to_dict(orient="records")}


def ablation(tables: dict, final: dict, chosen: dict) -> dict:
    out = {}
    for code, choice in final.items():
        family = choice["familia"]
        other = "completo" if chosen[family] == "sin_clima" else "sin_clima"
        t = tables[other][tables[other]["familia"] == family]
        common, _ = base.choose(t)
        oy = base.out_of_year(t, family)
        cells = oy[oy["municipality_code"] == code].sort_values("anio")
        exception = bool((cells["ll_propio"] < cells["ll_comun"]).all())
        name = base.choose(t[t["municipality_code"] == code])[0] if exception else common
        out[code] = {"familia": family, "conjunto": other, "candidato": name, "excepcion_hiperparametros": exception,
                     "dif_hiperparametros": (cells["ll_propio"] - cells["ll_comun"]).tolist()}
        print(f"  Ablación {code}: {family}, {other}, {name}{' (excepción)' if exception else ''}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canal", type=Path, required=True)
    parser.add_argument("--completo", type=Path, required=True)
    parser.add_argument("--sin-clima", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    folders = {"completo": args.completo, "sin_clima": args.sin_clima}
    print("Insumos (md5):")
    for s, folder in folders.items():
        for name in (base.TABLE, base.PREDICTIONS, base.MANIFEST):
            print(f"  {s}/{name}: {base.md5(folder / name)}")
        manifest = json.loads((folder / base.MANIFEST).read_text(encoding="utf-8"))
        print(f"  {s} busqueda_id: {manifest.get('busqueda_id', '(no está)')}")
    tables = {s: pd.read_parquet(f / base.TABLE) for s, f in folders.items()}
    predictions = {s: pd.read_parquet(f / base.PREDICTIONS) for s, f in folders.items()}
    baselines = pd.read_parquet(args.completo / base.BASELINE_PREDICTIONS)
    for s in SETS:
        print(f"\nIntegridad de la tabla ({s}):")
        base.check_table(tables[s], predictions[s], baselines)

    print("\nPaso 0 (contrato §6.6):")
    step0_result = {f: step0(tables, f) for f in base.FAMILIES}
    chosen = {f: r["conjunto"] for f, r in step0_result.items()}

    combined = pd.concat([tables[chosen[f]][tables[chosen[f]]["familia"] == f] for f in base.FAMILIES], ignore_index=True)
    combined_predictions = pd.concat([predictions[chosen[f]][predictions[chosen[f]]["familia"] == f]
                                      for f in base.FAMILIES], ignore_index=True)
    print("\nPasos 1-4 con el conjunto elegido de cada familia:")
    selection = base.select(combined)
    for code, choice in selection["municipios"].items():
        choice["conjunto"] = chosen[choice["familia"]]

    series = base.load_channel(args.canal)
    rows = base.development_rows(series)
    channel = base.channel_events(rows, series)
    print("\nτ (contrato §7.5):")
    tau = base.tau_step(rows, series, combined_predictions, selection["municipios"], channel)

    print("\nAblación (contrato §4.4 y §6.6):")
    ablation_result = ablation(tables, selection["municipios"], chosen)

    if args.json:
        payload = {"paso0": step0_result, "seleccion": selection, "tau": tau, "ablacion": ablation_result}
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nResultado escrito en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

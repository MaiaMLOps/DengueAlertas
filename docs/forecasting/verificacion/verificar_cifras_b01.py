"""Check every figure of the justification of rule B01 against the existing evidence (no new evaluation).

Reads only versioned evidence: reports/forecasting/evidence/complementario_20261120/ (b01_eventos.csv and
b01_resumen.json) and continuidad_20261120/continuidad_resumen.json. Recomputes the derived figures (false alarms per
municipality-year, fraction of weeks under alert, exact one-sided McNemar p-value) and compares each with the value
cited in docs/forecasting/analisis_complementario.md («Justificación de la regla B01 en el producto»). Exit code 1 if
any figure differs.

Run from the repository root:
    .venv/Scripts/python.exe -B docs/forecasting/verificacion/verificar_cifras_b01.py
"""

import json
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "reports" / "forecasting" / "evidence"
EVENTS = EVIDENCE / "complementario_20261120" / "b01_eventos.csv"
SUMMARY = EVIDENCE / "complementario_20261120" / "b01_resumen.json"
CONTINUITY = EVIDENCE / "continuidad_20261120" / "continuidad_resumen.json"

# (period, system) -> cited figures
CITED = {
    ("desarrollo", "regla_actual"): {"inicios": "8/10", "entradas": "21/44", "falsas_alarmas": 24, "fa_mun_anio": "1,00",
                                     "bajo_alerta": "47,0", "alertas": 43},
    ("desarrollo", "B01"): {"inicios": "10/10", "entradas": "38/44", "falsas_alarmas": 37, "fa_mun_anio": "1,54",
                            "bajo_alerta": "19,6", "alertas": 59},
    ("desarrollo", "canal"): {"inicios": "10/10", "entradas": "36/44", "falsas_alarmas": 48, "fa_mun_anio": "2,00",
                              "bajo_alerta": "32,5", "alertas": 65},
    ("prueba", "regla_actual"): {"inicios": "4/10", "entradas": "13/41", "falsas_alarmas": 18, "fa_mun_anio": "1,00",
                                 "bajo_alerta": "60,4", "alertas": 26, "anticipacion": "4,5/7,5"},
    ("prueba", "B01"): {"inicios": "9/10", "entradas": "24/41", "falsas_alarmas": 26, "fa_mun_anio": "1,44",
                        "bajo_alerta": "20,5", "alertas": 38, "anticipacion": "1,0/3,0"},
    ("prueba", "canal"): {"inicios": "8/10", "entradas": "19/41", "falsas_alarmas": 30, "fa_mun_anio": "1,67",
                          "bajo_alerta": "42,0", "alertas": 39, "anticipacion": "1,5/2,0"},
}
CITED_OTHER = {
    "municipio_anios_desarrollo": 24, "municipio_anios_prueba": 18,
    "discrepantes_prueba_B01_vs_regla_actual": "5/5", "p_mcnemar_prueba": "0,03125",
    "discrepantes_desarrollo_B01_vs_regla_actual": "2/2", "p_mcnemar_desarrollo": "0,25",
    "composicion_prueba": "268/272", "descarte_prueba": "196/464 (42,2 %)",
    "fa_tras_episodio_prueba": "11/26", "fa_tras_episodio_desarrollo": "13/37",
    "bss_continuidad_desarrollo": "0,252", "bss_continuidad_prueba": "0,136",
    "casillas_continuidad_prueba": "10/14",
}


def _es(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    events = pd.read_csv(EVENTS, dtype={"municipality_code": str})
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    continuity = json.loads(CONTINUITY.read_text(encoding="utf-8"))
    rows, failures = [], 0

    def check(name, cited, obtained):
        nonlocal failures
        ok = str(cited) == str(obtained)
        failures += not ok
        rows.append((name, cited, obtained, "sí" if ok else "NO"))

    years = {}
    for period in ("desarrollo", "prueba"):
        cells = events[(events["periodo"] == period) & (events["nivel"] == "municipio_anio")
                       & (events["sistema"] == "B01") & (events["semanas"] > 0)]
        years[period] = len(cells[["municipality_code", "anio"]].drop_duplicates())
        check(f"municipio-años, {period}", CITED_OTHER[f"municipio_anios_{period}"], years[period])
    for (period, system), cited in CITED.items():
        g = events[(events["periodo"] == period) & (events["sistema"] == system)
                   & (events["nivel"] == "global")].iloc[0]
        label = f"{period}, {system}"
        check(f"{label}: inicios avisados", cited["inicios"], f"{g.inicios_avisados}/{g.inicios_evaluables}")
        check(f"{label}: entradas avisadas", cited["entradas"], f"{g.entradas_avisadas}/{g.entradas_evaluables}")
        check(f"{label}: alertas", cited["alertas"], int(g.alertas))
        check(f"{label}: falsas alarmas", cited["falsas_alarmas"], int(g.falsas_alarmas))
        check(f"{label}: falsas alarmas por municipio-año", cited["fa_mun_anio"],
              _es(g.falsas_alarmas / years[period], 2))
        check(f"{label}: % de semanas bajo alerta", cited["bajo_alerta"],
              _es(100 * g.semanas_bajo_alerta / g.semanas, 1))
        if "anticipacion" in cited:
            check(f"{label}: anticipación mediana (contrato/primera)", cited["anticipacion"],
                  f"{_es(g.anticipacion_mediana_contrato, 1)}/{_es(g.anticipacion_mediana_primera, 1)}")

    for period in ("prueba", "desarrollo"):
        comp = summary["periodos"][period]["comparaciones"]["B01_vs_regla_actual"]
        check(f"discrepantes B01 frente a la regla actual, {period}",
              CITED_OTHER[f"discrepantes_{period}_B01_vs_regla_actual"], f"{comp['a_favor']}/{comp['discrepantes']}")
        p_value = float(binom.sf(comp["a_favor"] - 1, comp["discrepantes"], 0.5))
        check(f"p de McNemar exacto de una cola, {period}", CITED_OTHER[f"p_mcnemar_{period}"],
              f"{p_value:g}".replace(".", ","))
    composition = summary["periodos"]["prueba"]["composicion"]
    check("emisiones de B01 en zona de alerta, prueba", CITED_OTHER["composicion_prueba"],
          f"{composition['emisiones_B01_por_zona']['alerta']}/{composition['emisiones_B01']}")
    check("semanas en zona de alerta descartadas por p, prueba", CITED_OTHER["descarte_prueba"],
          f"{composition['semanas_zona_alerta_descartadas_por_p']}/{composition['semanas_zona_alerta']} "
          f"({_es(100 * composition['fraccion_descartada'], 1)} %)")
    for period in ("prueba", "desarrollo"):
        fa = continuity["periodos"][period]["falsas_alarmas_B01"]
        check(f"falsas alarmas de B01 tras un episodio, {period}", CITED_OTHER[f"fa_tras_episodio_{period}"],
              f"{fa['empiezan_tras_episodio']}/{fa['falsas_alarmas_B01']}")
        check(f"BSS de continuidad frente a B2, {period}", CITED_OTHER[f"bss_continuidad_{period}"],
              _es(continuity["periodos"][period]["estratos"]["zona_epidemia"]["bss_vs_b2"], 3))

    cells = continuity["periodos"]["prueba"]["estratos"]["zona_epidemia"]["casillas"]
    check("casillas de continuidad ganadas a B2 por Brier, prueba", CITED_OTHER["casillas_continuidad_prueba"],
          f"{cells['ganadas_brier']}/{cells['casillas']}")

    width = max(len(name) for name, *_ in rows)
    for name, cited, obtained, ok in rows:
        print(f"{name:<{width}}  citado={cited!s:<18} obtenido={obtained!s:<18} coincide={ok}")
    print(f"\n{len(rows) - failures} de {len(rows)} cifras coinciden.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

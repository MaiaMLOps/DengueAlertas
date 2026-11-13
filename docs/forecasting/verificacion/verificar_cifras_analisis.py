"""verificar_cifras_analisis.py: checks every evidence-derived figure of docs/forecasting/analisis_resultados.md.

Each check rebuilds, from the versioned evidence, the exact text that the analysis must contain (a full table row or a
prose fragment) and looks for it in the document. Values are read at full precision from the CSV/JSON files and rounded
once (ROUND_HALF_UP on the exact binary value), so double rounding (e.g. -0.4848 -> -0.485 -> -49 %) cannot pass.
It imports nothing from src/ and only reads files.

Usage (from the repository root):
    .venv/Scripts/python.exe -X utf8 -I docs/forecasting/verificacion/verificar_cifras_analisis.py
        [--analisis docs/forecasting/analisis_resultados.md] [--evidencia reports/forecasting/evidence]

Exit code 0 if every check passes; 1 otherwise. Each failure prints the check id and the expected text.
"""

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

MINUS = "−"
MUNIS = {
    "41001": "Neiva",
    "50001": "Villavicencio",
    "76001": "Cali",
    "08758": "Soledad",
    "54001": "Cúcuta",
    "08001": "Barranquilla",
}
ORDER = ["41001", "50001", "76001", "08758", "54001", "08001"]


# ------------------------------------------------------------------------------------------- formatting


def num(x, d, sign=False, pct=False):
    """Spanish decimal comma, U+2212 minus, one ROUND_HALF_UP rounding of the exact float."""
    v = Decimal(float(x)) * (100 if pct else 1)
    q = v.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    body = f"{abs(q):.{d}f}".replace(".", ",")
    if q < 0 or (q == 0 and v < 0 and sign):
        return MINUS + body
    return ("+" + body) if sign else body


def pct(x, d=0, sign=False):
    return num(x, d, sign=sign, pct=True) + " %"


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


# ------------------------------------------------------------------------------------------- evidence


class Ev:
    def __init__(self, root: Path):
        self.r = root

    def csv(self, rel, **kw):
        return pd.read_csv(self.r / rel, **kw)

    def json(self, rel):
        return json.loads((self.r / rel).read_text(encoding="utf-8"))


def code(s):
    return s.astype(str).str.zfill(5)


# ------------------------------------------------------------------------------------------- checks


def build_checks(ev: Ev):
    C = []

    def add(cid, text):
        C.append((cid, text))

    res = ev.json("prueba_20261112/resumen_prueba.json")
    ficha = ev.json("prueba_20261112/ficha_desempeno.json")
    g = ficha["global"]
    est = g["estimacion"]
    concl = res["conclusiones"]
    ev_g = g["eventos"]

    zc = ev.csv("exploratorio_20261112/zona_canal.csv")
    zl = ev.csv("exploratorio_20261112/zona_canal_lectura.csv")
    zi = ev.csv("exploratorio_20261112/zona_canal_intervalos.csv")
    um = ev.csv("exploratorio_20261112/clasificacion_umbrales.csv", dtype={"grupo": str})
    ct = ev.csv("exploratorio_20261112/clasificacion_curva_tau.csv")
    fa = ev.csv("exploratorio_20261112/fases.csv")
    se = ev.csv("exploratorio_20261112/semanas_desde_episodio.csv")
    em = ev.csv("exploratorio_20261112/estratos_municipio.csv")
    em["code"] = code(em["municipality_code"])
    ep = ev.csv("exploratorio_20261112/episodios_entrenamiento.csv")
    ep["code"] = code(ep["municipality_code"])
    ine = ev.csv("exploratorio_20261112/inicios_detalle.csv")
    tb = ev.csv("exploratorio_20261112/tasa_base_alerta.csv")
    mb2 = ev.csv("exploratorio_20261112/murphy_b2.csv")
    ape = ev.csv("exploratorio_20261112/ap_empates.csv")
    cu = ev.csv("exploratorio_20261112/contraste_u75.csv")
    mur = ev.csv("reportes_20261112/murphy.csv")
    cal = ev.csv("reportes_20261112/calibracion_anual.csv", dtype={"anio": str})
    hab = ev.csv("reportes_20261112/habilidad.csv")
    dvp = ev.csv("reportes_20261112/desarrollo_vs_prueba.csv")
    er = ev.csv("reportes_20261112/eventos_razon.csv", dtype={"grupo": str})
    du = ev.csv("reportes_20261112/duracion_alertas.csv", dtype={"grupo": str})
    conf = ev.csv("reportes_20261112/confiabilidad.csv", dtype={"grupo": str})
    amoc = ev.csv("reportes_20261112/curva_amoc.csv")
    ctau = ev.csv("reportes_20261112/clasificacion_tau.csv", dtype={"grupo": str})
    dcp = ev.csv("prueba_20261112/diag_casillas.csv", dtype={"municipality_code": str})
    dcd = ev.csv("diagnostico_desarrollo_sin_clima_20261112/diag_casillas.csv", dtype={"municipality_code": str})
    drp = ev.csv("prueba_20261112/diag_resumenes.csv", dtype={"grupo": str})
    drd = ev.csv("diagnostico_desarrollo_sin_clima_20261112/diag_resumenes.csv", dtype={"grupo": str})
    dtp = ev.csv("prueba_20261112/diag_transicion.csv", dtype={"anio": str})
    dtd = ev.csv("diagnostico_desarrollo_sin_clima_20261112/diag_transicion.csv", dtype={"anio": str})
    dep = ev.csv("prueba_20261112/diag_eventos.csv", dtype={"municipality_code": str})
    ded = ev.csv("diagnostico_desarrollo_sin_clima_20261112/diag_eventos.csv", dtype={"municipality_code": str})
    dap = ev.csv("prueba_20261112/diag_alertas.csv", dtype={"municipality_code": str})
    dsp = ev.csv("prueba_20261112/diag_serie.csv", dtype={"municipality_code": str})
    sel = ev.json("seleccion_conjuntos_20261112/resumen_seleccion.json")
    stau = ev.csv("seleccion_conjuntos_20261112/curva_tau.csv")

    def z(per, zona):
        return zc[(zc.periodo == per) & (zc.zona_t == zona)].iloc[0]

    def U(per, nivel, grupo, sistema, umbral):
        m = (um.periodo == per) & (um.nivel == nivel) & (um.grupo == grupo) & (um.sistema == sistema)
        if umbral is not None:
            m &= um.umbral == umbral
        return um[m].iloc[0]

    def ER(per, sis, grupo="todos"):
        return er[(er.periodo == per) & (er.sistema == sis) & (er.grupo == grupo)].iloc[0]

    def H(per, sis, ref):
        return hab[(hab.periodo == per) & (hab.sistema == sis) & (hab.referencia == ref)].iloc[0]

    # ---------------------------------------------------------------- header and Table 1
    add("H-480", f"{len(cu)} cantidades")
    assert cu["coincide"].all()
    b2c, b1c, abl = concl["vs_b2"], concl["vs_b1"], concl["ablacion_clima"]
    assert (b2c["nivel"], b1c["nivel"], abl["nivel"]) == ("tendencia favorable", "mejora consistente", "sin mejora")
    add(
        "T1-b2",
        row(
            "Modelo frente a B2 (principal)",
            "**Tendencia favorable**",
            f"BSS {num(est['bss_vs_b2'], 3, True)}; mejora de log-loss {pct(est['mejora_logloss_vs_b2'], 1, True)}",
            f"{b2c['casillas_ganadas_brier']} de 18; {b2c['casillas_ganadas_logloss']} de 18",
        ),
    )
    add(
        "T1-b1",
        row(
            "Modelo frente a B1",
            "**Mejora consistente**",
            f"BSS {num(est['bss_vs_b1'], 3, True)}; mejora de log-loss {pct(est['mejora_logloss_vs_b1'], 1, True)}",
            f"{b1c['casillas_ganadas_brier']} de 18; {b1c['casillas_ganadas_logloss']} de 18",
        ),
    )
    assert abl["nivel_brier"] == "sin mejora" and abl["nivel_logloss"] == "tendencia favorable"
    add(
        "T1-abl",
        row(
            "Ablación del clima (receta sin clima frente a la misma familia con clima)",
            "**Sin mejora** (con Brier, sin mejora; con log-loss, tendencia favorable; se adopta el menor, §8.4)",
            f"Brier {num(est['brier']['modelo'], 4)} (receta) frente a {num(est['brier']['ablacion'], 4)} (con clima); "
            f"log-loss {num(est['logloss']['modelo'], 3)} frente a {num(est['logloss']['ablacion'], 3)}",
            f"{abl['casillas_ganadas_brier']} de 18; {abl['casillas_ganadas_logloss']} de 18",
        ),
    )
    mc = ev_g["mcnemar"]
    add(
        "T1-ev",
        row(
            "Eventos frente al canal",
            "**Sin mejora**",
            f"Detección {ev_g['avisados_modelo']} de {ev_g['inicios_evaluables']} frente a {ev_g['avisados_canal']} de "
            f"{ev_g['inicios_evaluables']}; McNemar con {mc['discrepantes']} inicios discrepantes, "
            f"{mc['a_favor_modelo']} a favor del modelo",
            "—",
        ),
    )

    # ---------------------------------------------------------------- summary
    mt = mur[(mur.periodo == "prueba") & (mur.sistema == "modelo") & (mur.nivel == "global")].iloc[0]
    b2m = mb2[(mb2.periodo == "prueba") & (mb2.sistema == "B2") & mb2.agrupamiento.str.startswith("10")].iloc[0]
    tz = z("prueba", "todas")
    add("S1", f"(BSS {num(est['bss_vs_b2'], 3, True)}; log-loss {pct(est['mejora_logloss_vs_b2'], 1, True)})")
    add("S1-conf", f"confiabilidad {num(mt.confiabilidad, 3)} frente a {num(b2m.confiabilidad, 3)}")
    add(
        "S1-ap",
        f"AP {num(tz.ap_modelo, 2)} frente a {num(tz.ap_b2, 2)} de B2 y {num(tz.ap_persistencia, 2)} de la "
        f"persistencia; AUC {num(tz.auc_modelo, 2)} frente a {num(tz.auc_b2, 2)} y {num(tz.auc_persistencia, 2)}",
    )
    la = zl[(zl.periodo == "prueba") & (zl.zona_t == "alerta")].set_index("banda_p")
    le = zl[(zl.periodo == "prueba") & (zl.zona_t == "epidemia")].set_index("banda_p")
    add(
        "S2-alerta",
        f"el {pct(la.loc['[0.00, 0.05)', 'fraccion_de_la_zona'])} de esas semanas recibe p̂ < 0,05 y solo "
        f"el {pct(la.loc['[0.00, 0.05)', 'frecuencia_observada'])} llega a zona epidémica, frente al "
        f"{pct(la.loc['[0.20, 1.00)', 'frecuencia_observada'])} de las que reciben p̂ ≥ 0,20",
    )
    add(
        "S2-epi",
        f"el {pct(le.loc['[0.70, 1.00)', 'fraccion_de_la_zona'])} de las semanas recibe p̂ ≥ 0,70 y el "
        f"{pct(le.loc['[0.70, 1.00)', 'frecuencia_observada'])} de ellas sigue en zona epidémica 8 semanas "
        f"después; con p̂ < 0,30 solo sigue el {pct(le.loc['[0.00, 0.30)', 'frecuencia_observada'])}",
    )
    m30 = U("prueba", "global", "todos", "modelo", "max_f1_desarrollo")
    p30 = U("prueba", "global", "todos", "persistencia", None)
    add("S3", f"(F1 {num(m30.f1, 2)} y {num(p30.f1, 2)}; MCC {num(m30.mcc, 2)} y {num(p30.mcc, 2)})")
    fu = U("prueba", "estrato", "fuera", "modelo", "max_f1_desarrollo")
    de = U("prueba", "estrato", "dentro", "modelo", "max_f1_desarrollo")
    fud = U("desarrollo", "estrato", "fuera", "modelo", "max_f1_desarrollo")
    ded_ = U("desarrollo", "estrato", "dentro", "modelo", "max_f1_desarrollo")
    add(
        "S3-trans",
        f"recupera el {pct(fu.sensibilidad)} de las entradas a la zona epidémica y anticipa el "
        f"{pct(de.especificidad)} de las salidas",
    )
    fp_ = fa[fa.periodo == "prueba"].set_index("fase")
    add(
        "S4",
        f"aporta el {pct(fp_.loc['durante', 'porcentaje_mejora_neta'] / 100)} de la mejora neta del Brier frente a "
        f"B2 y las semanas tranquilas el {pct(fp_.loc['resto', 'porcentaje_mejora_neta'] / 100)}. Las 1–8 semanas "
        f"previas a un episodio aportan el {pct(fp_.loc['antes', 'porcentaje_mejora_neta'] / 100)} y las 1–13 semanas "
        f"posteriores restan el {pct(-fp_.loc['después', 'porcentaje_mejora_neta'] / 100)}",
    )
    mp, cp = ER("prueba", "modelo"), ER("prueba", "canal")
    add("S5", f"({mp.inicios_avisados} de 10 frente a {cp.inicios_avisados} de 10)")
    add("S5-fa", f"{pct(mp.proporcion_falsas_alarmas)} frente a {pct(cp.proporcion_falsas_alarmas)}")
    s13 = se[(se.periodo == "prueba") & (se.tramo == "1–13")].iloc[0]
    add("S-post", f"p̂ medio {num(s13.p_medio_modelo, 2)} frente a {pct(s13.prevalencia, 1)} observado [X]")
    add(
        "S-alerta",
        f"{pct(la.loc['[0.00, 0.05)', 'p_medio_modelo'])} predicho frente a "
        f"{pct(la.loc['[0.00, 0.05)', 'frecuencia_observada'])} observado [X]",
    )
    lost25 = int((~dcp[dcp.anio == 2025].gana_brier_b2).sum())
    assert lost25 == 4
    add("S-2025", "**2025** fue un año adverso en cuatro de seis municipios")

    # ---------------------------------------------------------------- Table 2.1
    for c in ORDER:
        f = ficha["municipios"][c]
        e = f["estimacion"]
        nv = {"favorable": "Favorable", "mixto": "Mixto", "desfavorable": "Desfavorable"}
        lvl_b2 = nv[f["estimacion"]["vs_b2"]["conclusion"]["nivel_descriptivo"]]
        lvl_b1 = nv[f["estimacion"]["vs_b1"]["conclusion"]["nivel_descriptivo"]]
        dev = dcd[dcd.municipality_code == c]
        tst = dcp[dcp.municipality_code == c]
        bss_dev = 1 - (dev.filas * dev.brier_modelo).sum() / (dev.filas * dev.brier_b2).sum()
        fe = f["eventos"]
        ini = (
            "sin inicios"
            if fe["inicios_evaluables"] == 0
            else f"{fe['avisados_modelo']} / {fe['avisados_canal']} de {fe['inicios_evaluables']}"
        )
        add(
            f"T2.1-{c}",
            row(
                MUNIS[c],
                lvl_b2,
                f"{num(e['bss_vs_b2'], 3, True)} ({num(bss_dev, 3, True)})",
                f"{int(tst.gana_brier_b2.sum())}/3 ({int(dev.gana_brier_b2.sum())}/4)",
                lvl_b1,
                f"{num(e['ap']['modelo'], 2)} / {num(e['ap']['B2'], 2)}",
                ini,
                f"{fe['entradas']['avisadas_modelo']} / {fe['entradas']['avisadas_canal']} de "
                f"{fe['entradas']['evaluables']}",
                f"{fe['falsas_alarmas']['modelo']} / {fe['falsas_alarmas']['canal']}",
            ),
        )

    # development level convention [X]: BSS > 0 and at least 3 of 4 wins, with Brier and log-loss, the lower adopted
    names = {2: "favorable", 1: "mixto", 0: "desfavorable"}
    dev_lvl = {}
    for c in ORDER:
        dv = dcd[dcd.municipality_code == c]
        bss = 1 - (dv.filas * dv.brier_modelo).sum() / (dv.filas * dv.brier_b2).sum()
        llm = 1 - (dv.filas * dv.logloss_modelo).sum() / (dv.filas * dv.logloss_b2).sum()
        lb = int(bss > 0) + int(dv.gana_brier_b2.sum() >= 3)
        ll = int(llm > 0) + int(dv.gana_logloss_b2.sum() >= 3)
        dev_lvl[MUNIS[c]] = names[min(lb, ll)]
    assert dev_lvl == {
        "Neiva": "favorable",
        "Villavicencio": "favorable",
        "Cali": "desfavorable",
        "Soledad": "mixto",
        "Cúcuta": "favorable",
        "Barranquilla": "favorable",
    }, dev_lvl
    add(
        "2-conv",
        "Con ella, Neiva, Villavicencio, Cúcuta y Barranquilla son favorables en desarrollo, Soledad es mixto y "
        "Cali es desfavorable",
    )
    assert all(ficha["municipios"][c]["eventos"]["mcnemar"]["nivel"] == "sin mejora" for c in ORDER)
    add("2-ev", "el nivel descriptivo por municipio de la ficha es «sin mejora» en los seis")

    # ---------------------------------------------------------------- Table 2.2 and episodes
    def emv(per, c, estr):
        return em[(em.periodo == per) & (em.code == c) & (em.estrato == estr)].iloc[0]

    for c in ORDER:
        cells = [
            f"{pct(emv('prueba', c, s).mejora_ll_b2, 0, True)} ({pct(emv('desarrollo', c, s).mejora_ll_b2, 0, True)})"
            for s in ("fuera", "dentro", "despues_1_13")
        ]
        add(f"T2.2-{c}", row(MUNIS[c], *cells))
    for c in ["50001", "08758", "41001", "08001", "54001", "76001"]:
        a = ep[(ep.periodo == "prueba") & (ep.code == c)].iloc[0]
        b = ep[(ep.periodo == "desarrollo") & (ep.code == c)].iloc[0]
        add(f"T2.3-{c}", row(MUNIS[c], a.episodios_antes_del_periodo, b.episodios_antes_del_periodo))

    # ---------------------------------------------------------------- municipality prose
    def cell(df, c, y):
        return df[(df.municipality_code == c) & (df.anio == y)].iloc[0]

    add("2.1-b", f"(BSS {num(cell(dcd, '41001', 2020).bss_b2, 2, True)})")
    nf = emv("prueba", "41001", "fuera")
    add(
        "2.1-c",
        f"{pct(nf.mejora_ll_b2, 0, True)} de log-loss en {nf.filas} emisiones, con {nf.positivas} positivas. "
        f"Dentro de ella mejora un {pct(emv('prueba', '41001', 'dentro').mejora_ll_b2, 0, True)}",
    )
    nd = emv("prueba", "41001", "despues_1_13")
    add("2.1-d", f"({pct(nd.mejora_ll_b2, 0, True)}, {nd.filas} semanas)")
    ne = ine[(ine.periodo == "prueba") & (ine.municipality_code == 41001) & (ine.sistema == "modelo")].set_index(
        "inicio"
    )
    add("2.1-e", f"encendida durante {ne.loc['2023-07-17', 'semanas_desde_tc']} semanas desde su confirmación")
    add("2.1-f", f"se había confirmado {ne.loc['2024-06-03', 'semanas_tc_a_inicio']} semanas antes")

    vt = cell(dcp, "50001", 2025)
    add("2.2-a", f"(BSS {num(vt.bss_b2, 3, True)}): un año epidémico con {vt.positivas} semanas positivas")
    vf, vd, vp = (emv("prueba", "50001", s).mejora_ll_b2 for s in ("fuera", "dentro", "despues_1_13"))
    add("2.2-b", f"({pct(vf, 0, True)} y {pct(vd, 0, True)}) y pierde después de los episodios ({pct(vp, 0, True)})")
    tv = ctau[(ctau.periodo == "prueba") & (ctau.grupo == "50001")].set_index("sistema")
    add(
        "2.2-c",
        f"su MCC es {num(tv.loc['modelo', 'mcc'], 2)} frente a {num(tv.loc['B2', 'mcc'], 2)} de B2, que marca "
        f"como positivas {tv.loc['B2', 'emisiones_positivas']} de {tv.loc['B2', 'filas']} semanas",
    )
    add(
        "2.2-d",
        f"el MCC del modelo ({num(U('prueba', 'municipio', '50001', 'modelo', 'max_f1_desarrollo').mcc, 2)}) "
        f"queda por debajo del de la persistencia "
        f"({num(U('prueba', 'municipio', '50001', 'persistencia', None).mcc, 2)})",
    )
    vi = ine[(ine.periodo == "prueba") & (ine.municipality_code == 50001) & (ine.sistema == "modelo")].iloc[0]
    add("2.2-e", f"se confirmó {vi.semanas_tc_a_inicio} semanas antes")
    add("2.2-f", f"se apagó {(pd.Timestamp(vi.inicio) - pd.Timestamp(vi.fin)).days // 7} semanas antes del inicio")

    cdv = dcd[dcd.municipality_code == "76001"]
    cpr = dcp[dcp.municipality_code == "76001"]
    add("2.3-a", f"con solo {cdv.positivas.sum()} semanas positivas en cuatro años")
    cin = em[(em.periodo == "prueba") & (em.code == "76001")].set_index("estrato")
    add(
        "2.3-b",
        f"dentro de la zona epidémica ({pct(cin.loc['dentro', 'mejora_ll_b2'], 0, True)}), donde están "
        f"{cin.loc['dentro', 'positivas']} de sus {cpr.positivas.sum()} semanas positivas; fuera de ella pierde "
        f"({pct(cin.loc['fuera', 'mejora_ll_b2'], 0, True)})",
    )
    add("2.3-c", f"aporta un BSS de {num(cell(dcp, '76001', 2025).bss_b2, 3, True)}")
    c23 = cell(dcp, "76001", 2023)
    add(
        "2.3-d",
        f"el Brier lo favorece ({num(c23.bss_b2, 3, True)}) y el log-loss no ({pct(c23.mejora_logloss_b2, 0, True)})",
    )
    add(
        "2.3-e",
        f"su MCC ({num(U('prueba', 'municipio', '76001', 'modelo', 'max_f1_desarrollo').mcc, 2)}) no supera al de "
        f"la persistencia ({num(U('prueba', 'municipio', '76001', 'persistencia', None).mcc, 2)})",
    )
    du_cali = du[(du.periodo == "prueba") & (du.sistema == "modelo") & (du.grupo == "76001")].iloc[0]
    add(
        "2.3-f",
        f"una sola alerta de {int(du_cali.duracion_max)} semanas",
    )

    sm = ficha["municipios"]["08758"]["estimacion"]
    add(
        "2.4-a",
        f"({num(sm['bss_vs_b2'], 3, True)}) y la mejora de log-loss, negativa "
        f"({pct(sm['mejora_logloss_vs_b2'], 1, True)})",
    )
    sdm, sdc = ER("desarrollo", "modelo", "08758"), ER("desarrollo", "canal", "08758")
    ts = ctau[(ctau.periodo == "prueba") & (ctau.grupo == "08758") & (ctau.sistema == "modelo")].iloc[0]
    add("2.4-c", f"el {pct(ts.emisiones_positivas / ts.filas)} de las semanas ({ts.emisiones_positivas} de {ts.filas})")
    tbq = ctau[(ctau.periodo == "prueba") & (ctau.grupo == "08001") & (ctau.sistema == "modelo")].iloc[0]
    add(
        "2.6-a",
        f"el {pct(tbq.emisiones_positivas / tbq.filas)} de las semanas ({tbq.emisiones_positivas} de {tbq.filas})",
    )
    shares = {
        c: ctau[(ctau.periodo == "prueba") & (ctau.grupo == c) & (ctau.sistema == "modelo")].iloc[0] for c in ORDER
    }
    assert max(shares, key=lambda c: shares[c].emisiones_positivas / shares[c].filas) == "08001"
    so = ine[(ine.periodo == "prueba") & (ine.municipality_code == 8758) & (ine.sistema == "modelo")].set_index(
        "inicio"
    )
    add(
        "2.4-d",
        f"una alerta confirmada {so.loc['2023-07-03', 'semanas_tc_a_inicio']} semanas antes "
        f"({so.loc['2023-07-03', 't_c']})",
    )
    add(
        "2.4-e",
        f"una alerta que se apagó "
        f"{(pd.Timestamp('2025-05-26') - pd.Timestamp(so.loc['2025-05-26', 'fin'])).days // 7} semanas antes",
    )

    cc = cell(dcp, "54001", 2023)
    add(
        "2.5-a",
        f"la pierde por Brier ({num(cc.bss_b2, 3, True)}) y la gana por log-loss "
        f"({pct(cc.mejora_logloss_b2, 0, True)})",
    )
    cu_ = ine[(ine.periodo == "prueba") & (ine.municipality_code == 54001) & (ine.sistema == "modelo")].set_index(
        "inicio"
    )
    add(
        "2.5-b",
        f"porque su alerta estaba confirmada desde {cu_.loc['2025-02-24', 'semanas_tc_a_inicio']} semanas antes",
    )
    add(
        "2.5-c",
        f"Pierde en las semanas posteriores a los episodios "
        f"({pct(emv('prueba', '54001', 'despues_1_13').mejora_ll_b2, 0, True)})",
    )

    bq = ficha["municipios"]["08001"]["estimacion"]
    add(
        "2.6-b",
        f"BSS {num(bq['bss_vs_b2'], 3, True)}, gana solo 2023 ({num(cell(dcp, '08001', 2023).bss_b2, 3, True)}) y "
        f"pierde 2024 ({num(cell(dcp, '08001', 2024).bss_b2, 3, True)}) y 2025 "
        f"({num(cell(dcp, '08001', 2025).bss_b2, 3, True)})",
    )
    add(
        "2.6-c",
        f"Pierde incluso dentro de la zona epidémica ({pct(emv('prueba', '08001', 'dentro').mejora_ll_b2, 0, True)})",
    )
    add("2.6-d", f"su AP es prácticamente la de B2 ({num(bq['ap']['modelo'], 2)} frente a {num(bq['ap']['B2'], 2)})")
    add("2.6-e", f"tras un 2022 con {cell(dcd, '08001', 2022).positivas} de 52 semanas en zona epidémica")

    # 2.7 patterns
    assert (
        sum(
            ficha["municipios"][c]["estimacion"]["ap"]["modelo"] > ficha["municipios"][c]["estimacion"]["ap"]["B2"]
            for c in ORDER
        )
        == 5
    )
    assert (em[em.estrato == "despues_1_13"].mejora_ll_b2 < 0).sum() == 12
    za, ze = z("prueba", "alerta"), z("prueba", "epidemia")
    assert (za.auc_modelo_mayor_municipios, ze.auc_modelo_mayor_municipios) == (6, 5)

    # ---------------------------------------------------------------- Table 3.1
    md_ = mur[(mur.periodo == "desarrollo") & (mur.sistema == "modelo") & (mur.nivel == "global")].iloc[0]
    b2d = mb2[(mb2.periodo == "desarrollo") & (mb2.sistema == "B2") & mb2.agrupamiento.str.startswith("10")].iloc[0]
    add(
        "T3.1-brier",
        row(
            "Brier",
            f"{num(mt.brier, 3)} ({num(md_.brier, 3)})",
            f"{num(b2m.brier, 3)} ({num(b2d.brier, 3)})",
            "Menor es mejor",
        ),
    )
    add("T3.1-ll", row("Log-loss", num(est["logloss"]["modelo"], 3), num(est["logloss"]["B2"], 3), "Menor es mejor"))
    add(
        "T3.1-rel",
        f"| {num(mt.confiabilidad, 3)} ({num(md_.confiabilidad, 3)}) | {num(b2m.confiabilidad, 3)} "
        f"({num(b2d.confiabilidad, 3)}) [X] |",
    )
    add(
        "T3.1-res",
        f"| {num(mt.resolucion, 3)} ({num(md_.resolucion, 3)}) | {num(b2m.resolucion, 3)} "
        f"({num(b2d.resolucion, 3)}) [X] |",
    )
    cp_ = cal[cal.periodo == "prueba"].set_index("anio")
    add(
        "T3.1-cal",
        " / ".join(num(cp_.loc[y, "calibracion_modelo"], 3, True) for y in ("2023", "2024", "2025"))
        + " | "
        + " / ".join(num(cp_.loc[y, "calibracion_B2"], 3, True) for y in ("2023", "2024", "2025")),
    )
    zd = z("desarrollo", "todas")
    add(
        "T3.1-nit",
        f"| {num(tz.desv_p_modelo, 2)} ({num(zd.desv_p_modelo, 2)}) | {num(tz.desv_p_b2, 2)} "
        f"({num(zd.desv_p_b2, 2)}) [X] |",
    )
    add("3.1-unc", f"incertidumbre es la misma para ambos ({num(mt.incertidumbre, 3)})")
    cg = conf[(conf.periodo == "prueba") & (conf.sistema == "modelo") & (conf.nivel == "global")].sort_values(
        "intervalo"
    )
    lo = cg.iloc[(cg.p_media - 0.05).abs().argmin()]
    hi = cg.iloc[-1]
    add(
        "3.1-ext",
        f"con p̂ medio ≈ {num(lo.p_media, 2)} la frecuencia observada es {pct(lo.frecuencia_observada)}, y en el "
        f"de p̂ ≈ {num(hi.p_media, 2)} es {pct(hi.frecuencia_observada)}",
    )

    # ---------------------------------------------------------------- Table 3.2 and PR
    dvz = z("desarrollo", "todas")
    add(
        "T3.2-todas",
        row(
            "Todas",
            f"{num(tz.prevalencia, 2)} ({num(dvz.prevalencia, 2)})",
            f"{num(tz.ap_modelo, 2)} ({num(dvz.ap_modelo, 2)})",
            f"{num(tz.ap_b2, 2)} ({num(dvz.ap_b2, 2)})",
            f"{num(tz.ap_persistencia, 2)} ({num(dvz.ap_persistencia, 2)})",
            f"{num(tz.auc_modelo, 2)} ({num(dvz.auc_modelo, 2)})",
            f"{num(tz.auc_b2, 2)} ({num(dvz.auc_b2, 2)})",
            f"{num(tz.auc_persistencia, 2)} ({num(dvz.auc_persistencia, 2)})",
        ),
    )
    tf, tdf = (
        dtp[(dtp.anio == "todos") & (dtp.estrato == "fuera")].iloc[0],
        dtd[(dtd.anio == "todos") & (dtd.estrato == "fuera")].iloc[0],
    )
    tn, tdn = (
        dtp[(dtp.anio == "todos") & (dtp.estrato == "dentro")].iloc[0],
        dtd[(dtd.anio == "todos") & (dtd.estrato == "dentro")].iloc[0],
    )
    add(
        "T3.2-fuera",
        row(
            "Fuera de zona epidémica en t",
            f"{num(tf.positivas / tf.filas, 2)} ({num(tdf.positivas / tdf.filas, 2)})",
            f"{num(tf.ap_modelo, 2)} ({num(tdf.ap_modelo, 2)})",
            f"{num(tf.ap_b2, 2)} ({num(tdf.ap_b2, 2)})",
            "—",
            "—",
            "—",
            "—",
        ),
    )
    ze_d = z("desarrollo", "epidemia")
    add(
        "T3.2-dentro",
        row(
            "Dentro de zona epidémica en t",
            f"{num(tn.positivas / tn.filas, 2)} ({num(tdn.positivas / tdn.filas, 2)})",
            f"{num(tn.ap_modelo, 2)} ({num(tdn.ap_modelo, 2)})",
            f"{num(tn.ap_b2, 2)} ({num(tdn.ap_b2, 2)})",
            "—",
            f"{num(ze.auc_modelo, 2)} ({num(ze_d.auc_modelo, 2)})",
            f"{num(ze.auc_b2, 2)} ({num(ze_d.auc_b2, 2)})",
            "—",
        ),
    )
    # precision-recall points of the model in test, from the saved test series
    y = dsp.y.to_numpy()
    p = dsp.p_modelo.to_numpy()
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    last = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]  # last index of each tied group
    tp = np.cumsum(ys)[last]
    prec, rec = tp / (last + 1), tp / y.sum()
    # recall up to which precision stays >= 0.80 without interruption (ignoring the first 5 % of recall)
    first_below = np.where((prec < 0.80) & (rec > 0.05))[0][0]
    rec80 = rec[first_below - 1]
    floor55 = float(prec[rec <= 0.5528].min())  # lowest precision up to recall 0.55
    assert floor55 >= 0.78, floor55
    add(
        "3.2-pr",
        f"precisión de 0,80 o más hasta una sensibilidad de {num(np.floor(rec80 * 100) / 100, 2)} y de al menos "
        f"{num(np.floor(floor55 * 100) / 100, 2)} hasta 0,55, frente a {num(p30.vpp, 2)} de la persistencia",
    )
    add("3.2-ap", f"la AP es casi la misma ({num(dvz.ap_modelo, 2)} y {num(tz.ap_modelo, 2)})")
    add(
        "3.2-estr",
        f"fuera de la zona epidémica ({num(tf.ap_modelo, 2)} frente a {num(tf.ap_b2, 2)}) y la supera dentro "
        f"({num(tn.ap_modelo, 2)} frente a {num(tn.ap_b2, 2)})",
    )

    # ---------------------------------------------------------------- 3.3
    b2r = U("prueba", "global", "todos", "B2", "receta_0.05")
    add("3.3-89", f"B2 marca el {pct(b2r.fraccion_emitida)} de las semanas")
    cb = ct[(ct.periodo == "prueba") & (ct.sistema == "B2")]
    cpers = U("prueba", "global", "todos", "persistencia", None).emisiones_positivas
    same = cb[cb.emisiones_positivas == cpers].tau.round(2)
    add("3.3-range", f"para cualquier umbral entre {num(same.min(), 2)} y {num(same.max(), 2)}")
    add("3.3-340", f"(marca las mismas {int(cpers)} semanas)")

    def r33(label, sist, umbral, total=False):
        a = U("prueba", "global", "todos", sist, umbral)
        b = U("desarrollo", "global", "todos", sist, umbral)
        return row(
            label,
            f"{pct(a.fraccion_emitida)} ({pct(b.fraccion_emitida)})",
            f"{num(a.sensibilidad, 2)} ({num(b.sensibilidad, 2)})",
            f"{num(a.especificidad, 2)} ({num(b.especificidad, 2)})",
            f"{num(a.vpp, 2)} ({num(b.vpp, 2)})",
            f"{num(a.vpn, 2)} ({num(b.vpn, 2)})",
            f"{num(a.f1, 2)} ({num(b.f1, 2)})",
            f"{num(a.mcc, 2)} ({num(b.mcc, 2)})",
        )

    add("T3.3-m05", r33("Modelo, τ = 0,05 (receta)", "modelo", "receta_0.05"))
    add("T3.3-b05", r33("B2, τ = 0,05", "B2", "receta_0.05"))
    add("T3.3-m30", r33("Modelo, τ = 0,30 (F1 máxima en desarrollo)", "modelo", "max_f1_desarrollo"))
    add("T3.3-m50", r33("Modelo, τ = 0,50", "modelo", "sin_seleccion_0.50"))
    add("T3.3-pers", r33("Persistencia (= B2 con τ = 0,30)", "persistencia", None))
    a = U("prueba", "global", "todos", "modelo", "receta_0.05")
    b = U("desarrollo", "global", "todos", "modelo", "receta_0.05")
    add(
        "T3.3-todas",
        row(
            "Marcar todas las semanas",
            "100 %",
            "1,00",
            "0,00",
            f"{num(a.positivas / a.filas, 2)} ({num(b.positivas / b.filas, 2)})",
            "—",
            f"{num(a.f1_referencia, 2)} ({num(b.f1_referencia, 2)})",
            "—",
        ),
    )
    add("3.3-mcc05", f"(MCC {num(a.mcc, 2)} frente a {num(b2r.mcc, 2)})")
    m30d = U("desarrollo", "global", "todos", "modelo", "max_f1_desarrollo")
    p30d = U("desarrollo", "global", "todos", "persistencia", None)
    add(
        "3.3-dev",
        f"(en desarrollo, {num(m30d.f1, 2)} frente a {num(p30d.f1, 2)} y {num(m30d.mcc, 2)} frente a "
        f"{num(p30d.mcc, 2)})",
    )
    win = [
        MUNIS[c]
        for c in ORDER
        if U("prueba", "municipio", c, "modelo", "max_f1_desarrollo").mcc
        > U("prueba", "municipio", c, "persistencia", None).mcc
    ]
    assert win == ["Neiva", "Soledad"], win
    worse_years = [
        yy
        for yy in ("2023", "2024", "2025")
        if U("prueba", "anio", yy, "modelo", "max_f1_desarrollo").mcc
        < U("prueba", "anio", yy, "persistencia", None).mcc
    ]
    assert worse_years == ["2023", "2025"], worse_years

    # Table 3.4
    def share(r):
        return r.emisiones_positivas / r.filas

    add(
        "T3.4-fuera",
        row(
            "Fuera de zona epidémica en t",
            f"{fu.filas} ({fu.positivas}) ({fud.filas} ({fud.positivas}))",
            f"Marca el {pct(share(fu))} ({pct(share(fud))}) de las semanas; recupera el {pct(fu.sensibilidad)} "
            f"({pct(fud.sensibilidad)}) de las entradas con precisión {num(fu.vpp, 2)} ({num(fud.vpp, 2)}), frente a "
            f"una tasa base de {num(fu.positivas / fu.filas, 2)} ({num(fud.positivas / fud.filas, 2)})",
            "Recupera el 0 % de las entradas, por construcción",
        ),
    )
    neg, negd = de.vn + de.fn, ded_.vn + ded_.fn
    add(
        "T3.4-dentro",
        row(
            "Dentro de zona epidémica en t",
            f"{de.filas} ({de.positivas}) ({ded_.filas} ({ded_.positivas}))",
            f"Predice la salida en el {pct(neg / de.filas)} ({pct(negd / ded_.filas)}) de las semanas; acierta en "
            f"el {pct(de.vpn)} ({pct(ded_.vpn)}) de esas predicciones, frente a una tasa base de salida de "
            f"{num((de.filas - de.positivas) / de.filas, 2)} ({num((ded_.filas - ded_.positivas) / ded_.filas, 2)}), "
            f"y anticipa el {pct(de.especificidad)} ({pct(ded_.especificidad)}) de las salidas",
            "Anticipa el 0 % de las salidas, por construcción",
        ),
    )
    ratios = [
        fu.vpp / (fu.positivas / fu.filas),
        fud.vpp / (fud.positivas / fud.filas),
        de.vpn / ((de.filas - de.positivas) / de.filas),
        ded_.vpn / ((ded_.filas - ded_.positivas) / ded_.filas),
    ]
    lo_r, hi_r = int(np.floor(min(ratios))), int(np.ceil(max(ratios)))
    words = {2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}
    add(
        "3.3-ratio",
        f"con una precisión de {words[lo_r]} a {words[hi_r - 1] if hi_r - 1 > lo_r else words[hi_r]} veces la "
        f"tasa base",
    )
    ntr, ntr_ok = fu.positivas + (de.filas - de.positivas), fu.vp + de.vn
    add("3.3-vol", f"en prueba anticipa {ntr_ok} de {ntr} transiciones")
    assert 1 / 5.5 < ntr_ok / ntr < 1 / 5

    # ---------------------------------------------------------------- Table 3.5
    def r35(zona, label):
        a, b = z("prueba", zona), z("desarrollo", zona)
        ap = "sin positivas" if np.isnan(a.ap_modelo) else f"{num(a.ap_modelo, 2)} / {num(a.ap_b2, 2)}"
        au = "sin positivas" if np.isnan(a.auc_modelo) else f"{num(a.auc_modelo, 2)} / {num(a.auc_b2, 2)}"
        cnt = (
            "—"
            if a.municipios_con_ambas_clases == 0
            else f"{a.auc_modelo_mayor_municipios} de {a.municipios_con_ambas_clases}"
        )
        return row(
            label,
            f"{a.filas} ({b.filas})",
            f"{num(a.prevalencia, 2)} ({num(b.prevalencia, 2)})",
            f"{pct(a.mejora_ll_b2, 0, True)} ({pct(b.mejora_ll_b2, 0, True)})",
            f"{num(a.bss_b2, 2, True)} ({num(b.bss_b2, 2, True)})",
            f"{ap} ({num(b.ap_modelo, 2)} / {num(b.ap_b2, 2)})",
            f"{au} ({num(b.auc_modelo, 2)} / {num(b.auc_b2, 2)})",
            f"{cnt} ({b.auc_modelo_mayor_municipios} de {b.municipios_con_ambas_clases})",
        )

    add("T3.5-seg", r35("seguridad", "Seguridad"))
    add("T3.5-ale", r35("alerta", "Alerta"))
    add("T3.5-epi", r35("epidemia", "Epidemia"))
    add("3.4-auc", f"(AUC {num(za.auc_modelo, 2)} frente a {num(za.auc_b2, 2)})")
    zia = zi[(zi.periodo == "prueba") & (zi.zona_t == "alerta")]
    hi_rows = zia[(zia.p_min >= 0.5) & (zia.p_min < 0.9)]
    add(
        "3.4-cal",
        f"con p̂ entre 0,5 y 0,9, es {num(hi_rows.frecuencia_observada.min() * 100, 0)}–"
        f"{pct(hi_rows.frecuencia_observada.max())} y no {num(hi_rows.p_medio_modelo.min() * 100, 0)}–"
        f"{num(hi_rows.p_medio_modelo.max() * 100, 0)} %",
    )
    add("3.4-ll", f"la mejora de log-loss en esta zona es solo {pct(za.mejora_ll_b2, 0, True)} en prueba")
    sg = zl[(zl.periodo == "prueba") & (zl.zona_t == "seguridad")].set_index("banda_p")
    add("3.4-seg", f"{sg.loc['[0.00, 0.05)', 'filas']} de las {z('prueba', 'seguridad').filas} semanas tienen p̂ < 0,05")
    add("3.4-segb2", f"B2 asigna en promedio {num(z('prueba', 'seguridad').p_medio_b2, 2)}")

    # ---------------------------------------------------------------- Table 3.6 and 3.5 prose
    def r36(fase, label):
        a = fa[(fa.periodo == "prueba") & (fa.fase == fase)].iloc[0]
        b = fa[(fa.periodo == "desarrollo") & (fa.fase == fase)].iloc[0]
        return row(
            label,
            f"{a.filas} ({b.filas})",
            f"{pct(a.porcentaje_mejora_neta / 100, 0, True)} ({pct(b.porcentaje_mejora_neta / 100, 0, True)})",
            f"{pct(a.mejora_ll_b2, 1 if abs(a.mejora_ll_b2) < 0.2 else 0, True)} ({pct(b.mejora_ll_b2, 0, True)})",
            num(a.bss_b2, 2, True),
            f"{a.municipios_ganan_ll} de 6 ({b.municipios_ganan_ll} de 6)",
        )

    add("T3.6-antes", r36("antes", "Antes (1–8 semanas previas)"))
    add("T3.6-dur", r36("durante", "Durante el episodio"))
    add("T3.6-des", r36("después", "Después (1–13 semanas)"))
    add("T3.6-res", r36("resto", "Resto (tranquilas)"))
    fm = ev.csv("exploratorio_20261112/fases_municipio.csv")
    neg_after = int((fm[fm.fase == "después"].mejora_ll_b2 < 0).sum())
    add("3.5-cells", f"«después» es rojo en {neg_after} de 12 celdas")
    s13d = se[(se.periodo == "desarrollo") & (se.tramo == "1–13")].iloc[0]
    add("3.5-win", f"tiene {s13.filas} semanas objetivo en prueba, y no las {fp_.loc['después', 'filas']} de la fase")
    add(
        "3.5-cal",
        f"el p̂ medio del modelo es {num(s13.p_medio_modelo, 2)} y el de B2 {num(s13.p_medio_b2, 2)}, mientras que "
        f"la frecuencia observada es {num(s13.prevalencia, 3)}; en desarrollo, {num(s13d.p_medio_modelo, 2)}, "
        f"{num(s13d.p_medio_b2, 2)} y {num(s13d.prevalencia, 3)}",
    )
    add(
        "3.5-ll",
        f"la mejora de log-loss es {pct(s13.mejora_ll_b2, 0, True)} (desarrollo, {pct(s13d.mejora_ll_b2, 0, True)})",
    )
    add("3.5-sal", f"el modelo anticipa algunas salidas ({pct(de.especificidad)} en prueba)")

    # ---------------------------------------------------------------- Table 4.1 and section 4
    def r41(zona, banda, label_z, label_b, first):
        a = zl[(zl.periodo == "prueba") & (zl.zona_t == zona) & (zl.banda_p == banda)].iloc[0]
        b = zl[(zl.periodo == "desarrollo") & (zl.zona_t == zona) & (zl.banda_p == banda)].iloc[0]
        obs = f"{a.positivas} de {a.filas} semanas" if a.filas < 10 else pct(a.frecuencia_observada, 1)
        zone = f"{pct(a.frecuencia_de_la_zona, 1)} ({pct(b.frecuencia_de_la_zona, 1)})" if first else ""
        return row(
            label_z,
            label_b,
            f"{pct(a.fraccion_de_la_zona)} ({pct(b.fraccion_de_la_zona)})",
            f"{obs} ({pct(b.frecuencia_observada, 1)})",
            zone,
            f"{num(a.p_medio_b2, 2)} ({num(b.p_medio_b2, 2)})",
        )

    add("T4.1-s1", r41("seguridad", "[0.00, 0.05)", "Seguridad", "< 0,05", True))
    add("T4.1-s2", r41("seguridad", "[0.05, 1.00)", "Seguridad", "≥ 0,05", False).replace("|  |", "| |"))
    add("T4.1-a1", r41("alerta", "[0.00, 0.05)", "Alerta", "< 0,05", True))
    add("T4.1-a2", r41("alerta", "[0.05, 0.20)", "Alerta", "0,05–0,20", False).replace("|  |", "| |"))
    add("T4.1-a3", r41("alerta", "[0.20, 1.00)", "Alerta", "≥ 0,20", False).replace("|  |", "| |"))
    add("T4.1-e1", r41("epidemia", "[0.00, 0.30)", "Epidemia", "< 0,30", True))
    add("T4.1-e2", r41("epidemia", "[0.30, 0.70)", "Epidemia", "0,30–0,70", False).replace("|  |", "| |"))
    add("T4.1-e3", r41("epidemia", "[0.70, 1.00)", "Epidemia", "≥ 0,70", False).replace("|  |", "| |"))
    add(
        "4-u1",
        f"En prueba, el {pct(le.loc['[0.70, 1.00)', 'fraccion_de_la_zona'])} de esas semanas tuvo p̂ ≥ 0,70 y el "
        f"{pct(le.loc['[0.70, 1.00)', 'frecuencia_observada'])} de ellas seguía en zona epidémica 8 semanas después; "
        f"con p̂ < 0,30, solo el {pct(le.loc['[0.00, 0.30)', 'frecuencia_observada'])}",
    )
    dm05 = U("prueba", "estrato", "dentro", "modelo", "receta_0.05")
    add("4-u1b", f"el modelo emite positivo en el {pct(dm05.fraccion_emitida, 1)} de las semanas en zona epidémica")
    add("4-u2", f"solo el {pct(za.prevalencia)} termina en zona epidémica")
    add(
        "4-u2b",
        f"el {pct(la.loc['[0.00, 0.05)', 'fraccion_de_la_zona'])} de esas semanas, con p̂ < 0,05, tuvo un "
        f"{pct(la.loc['[0.00, 0.05)', 'frecuencia_observada'])} de conversión; el "
        f"{pct(la.loc['[0.20, 1.00)', 'fraccion_de_la_zona'])}, con p̂ ≥ 0,20, tuvo un "
        f"{pct(la.loc['[0.20, 1.00)', 'frecuencia_observada'])}",
    )
    sd = zl[(zl.periodo == "desarrollo") & (zl.zona_t == "seguridad")].set_index("banda_p")
    add(
        "4-u3",
        f"se acompañó de {pct(sg.loc['[0.00, 0.05)', 'frecuencia_observada'])} de semanas epidémicas en prueba y "
        f"{pct(sd.loc['[0.00, 0.05)', 'frecuencia_observada'], 1)} en desarrollo",
    )
    dup = du[(du.periodo == "prueba") & (du.sistema == "modelo") & (du.grupo == "todos")].iloc[0]
    add("4-no", f"la alerta del modelo está encendida el {pct(dup.fraccion_bajo_alerta)} del tiempo")

    # ---------------------------------------------------------------- Table 5.1
    md, cd = ER("desarrollo", "modelo"), ER("desarrollo", "canal")
    add(
        "T5.1-ini",
        row(
            "Inicios avisados (detección)",
            f"{mp.inicios_avisados} de 10 ({md.inicios_avisados} de 10)",
            f"{cp.inicios_avisados} de 10 ({cd.inicios_avisados} de 10)",
        ),
    )
    add(
        "T5.1-mcn",
        row(
            "McNemar sobre inicios discrepantes",
            f"{mc['discrepantes']} discrepantes, {mc['a_favor_modelo']} a favor del modelo: sin mejora",
            "—",
        ),
    )

    def ee(r):
        return int(r.entradas_evaluables)

    add(
        "T5.1-ent",
        row(
            "Entradas avisadas",
            f"{mp.entradas_avisadas} de {ee(mp)} ({md.entradas_avisadas} de {ee(md)})",
            f"{cp.entradas_avisadas} de {ee(cp)} ({cd.entradas_avisadas} de {ee(cd)})",
        ),
    )
    add(
        "T5.1-juz",
        row(
            "Alertas juzgables",
            f"{mp.alertas_juzgables} ({md.alertas_juzgables})",
            f"{cp.alertas_juzgables} ({cd.alertas_juzgables})",
        ),
    )
    add(
        "T5.1-cont",
        row(
            "Alertas de continuidad (no juzgan inicios)",
            f"{mp.alertas_continuidad} ({md.alertas_continuidad})",
            f"{cp.alertas_continuidad} ({cd.alertas_continuidad})",
        ),
    )
    add(
        "T5.1-fa",
        row(
            "Falsas alarmas", f"{mp.falsas_alarmas} ({md.falsas_alarmas})", f"{cp.falsas_alarmas} ({cd.falsas_alarmas})"
        ),
    )
    add(
        "T5.1-fam",
        row(
            "Falsas alarmas por municipio-año",
            f"{num(mp.falsas_por_municipio_anio, 2)} ({num(md.falsas_por_municipio_anio, 2)})",
            f"{num(cp.falsas_por_municipio_anio, 2)} ({num(cd.falsas_por_municipio_anio, 2)})",
        ),
    )
    add(
        "T5.1-prop",
        row(
            "Proporción de alertas juzgables que son falsas",
            f"{pct(mp.proporcion_falsas_alarmas)} ({pct(md.proporcion_falsas_alarmas)})",
            f"{pct(cp.proporcion_falsas_alarmas)} ({pct(cd.proporcion_falsas_alarmas)})",
        ),
    )
    add(
        "T5.1-ant",
        row(
            "Anticipación mediana en los inicios avisados",
            f"{num(mp.anticipacion_mediana, 1)} semanas ({num(md.anticipacion_mediana, 1)})",
            f"{num(cp.anticipacion_mediana, 1)} semanas ({num(cd.anticipacion_mediana, 1)})",
        ),
    )
    dcn = du[(du.periodo == "prueba") & (du.sistema == "canal") & (du.grupo == "todos")].iloc[0]
    add(
        "T5.1-dur",
        row(
            "Duración mediana de la alerta, desde t_c (p75; máxima)",
            f"{num(dup.duracion_mediana, 0)} semanas ({num(dup.duracion_p75, 0)}; {num(dup.duracion_max, 0)})",
            f"{num(dcn.duracion_mediana, 0)} semanas ({num(dcn.duracion_p75, 0)}; {num(dcn.duracion_max, 0)})",
        ),
    )
    dud = du[(du.periodo == "desarrollo") & (du.sistema == "modelo") & (du.grupo == "todos")].iloc[0]
    dcd_ = du[(du.periodo == "desarrollo") & (du.sistema == "canal") & (du.grupo == "todos")].iloc[0]
    add(
        "T5.1-baj",
        row(
            "Semanas de emisión bajo alerta",
            f"{pct(dup.fraccion_bajo_alerta)} ({pct(dud.fraccion_bajo_alerta)})",
            f"{pct(dcn.fraccion_bajo_alerta)} ({pct(dcd_.fraccion_bajo_alerta)})",
        ),
    )
    tbp = tb[(tb.periodo == "prueba") & (tb.sistema == "modelo")].iloc[0]
    tbd = tb[(tb.periodo == "desarrollo") & (tb.sistema == "modelo")].iloc[0]
    add(
        "T5.1-tb1",
        row(
            "Inicios con el sistema bajo alerta en la semana del inicio [X]",
            f"{tbp.inicios_bajo_alerta} de 10 ({tbd.inicios_bajo_alerta} de 10)",
            "—",
        ),
    )
    add(
        "T5.1-tb2",
        row(
            "Esperados por la tasa base de semanas bajo alerta [X]",
            f"{num(tbp.esperado_por_tasa_base, 1)} ({num(tbd.esperado_por_tasa_base, 1)})",
            "—",
        ),
    )
    am = amoc[(amoc.periodo == "prueba") & (amoc.sistema == "modelo")]
    assert am.inicios_avisados.max() == 4
    # Table 5.2 and the fails
    fails = ine[
        (ine.periodo == "prueba")
        & (ine.sistema == "modelo")
        & (ine.estado == "alerta confirmada más de 12 semanas antes")
    ]
    fa_flags = []
    for _, r_ in fails.iterrows():
        a_ = dap[
            (dap.sistema == "modelo")
            & (dap.municipality_code == f"{r_.municipality_code:05d}")
            & (dap.confirmacion == r_.t_c)
        ].iloc[0]
        fa_flags.append(bool(a_.falsa_alarma))
    assert len(fails) == 3 and sum(fa_flags) == 2, (len(fails), fa_flags)
    add(
        "5-fails",
        "En tres, el modelo tenía una alerta encendida en la semana del inicio, pero confirmada más de 12 "
        "semanas antes, así que según §7.4 no lo avisa",
    )
    so_fa = dap[(dap.sistema == "modelo") & (dap.falsa_alarma)].confirmacion.tolist()
    assert "2022-11-14" in so_fa and "2024-11-11" in so_fa and "2023-07-03" not in so_fa
    add("5-tb", f"estaría bajo alerta en unos {num(tbp.esperado_por_tasa_base, 0)} de 10")
    add(
        "5-tbd",
        f"En desarrollo fueron {tbd.inicios_bajo_alerta} de 10 frente a {num(tbd.esperado_por_tasa_base, 1)} esperados",
    )
    fpv = fa[fa.periodo == "prueba"].set_index("fase").loc["antes"]
    fdv = fa[fa.periodo == "desarrollo"].set_index("fase").loc["antes"]
    add("5-antes", f"{pct(fpv.mejora_ll_b2, 1, True)} en prueba y {pct(fdv.mejora_ll_b2, 0, True)} en desarrollo")

    # ---------------------------------------------------------------- Table 6.1
    reg_abbr = {"epidemica": "E", "post_epidemica": "P", "tranquila": "T"}
    alld = pd.concat([dcd.assign(per="desarrollo"), dcp.assign(per="prueba")])
    allr = pd.concat([drd, drp])
    alle = pd.concat([ded, dep])
    for yy in range(2019, 2026):
        cells_ = alld[alld.anio == yy]
        cnt = cells_.regimen.value_counts()
        reg = ", ".join(f"{cnt[k]} {reg_abbr[k]}" for k in ("epidemica", "post_epidemica", "tranquila") if k in cnt)
        rs = allr[(allr.nivel == "anio") & (allr.grupo == str(yy))].iloc[0]
        cy = cal[cal.anio == str(yy)].iloc[0]
        per = "desarrollo" if yy <= 2022 else "prueba"
        mm = U(per, "anio", str(yy), "modelo", "max_f1_desarrollo").mcc
        pm = U(per, "anio", str(yy), "persistencia", None).mcc
        on = alle[(alle.tipo == "inicio") & (alle.evaluable) & (alle.anio == yy)]
        add(
            f"T6.1-{yy}",
            row(
                yy,
                reg,
                int(cy.positivas),
                num(rs.bss_b2_filas, 3, True),
                f"{rs.victorias_brier_b2} / {rs.victorias_brier_b1}",
                f"{num(cy.calibracion_modelo, 3, True)} / {num(cy.calibracion_B2, 3, True)}",
                f"{num(mm, 2)} / {num(pm, 2)}",
                f"{int(on.avisado_modelo.sum())} / {int(on.avisado_canal.sum())} de {len(on)}",
            ),
        )
    add("6-2023", f"prevalencia alta ({pct(cal[cal.anio == '2023'].iloc[0].prevalencia, 1)})")
    add("6-2025", f"({pct(cal[cal.anio == '2025'].iloc[0].prevalencia, 1)})")
    m25 = mur[(mur.periodo == "prueba") & (mur.sistema == "modelo") & (mur.grupo == "2025")].iloc[0]
    m23 = mur[(mur.periodo == "prueba") & (mur.sistema == "modelo") & (mur.grupo == "2023")].iloc[0]
    m24 = mur[(mur.periodo == "prueba") & (mur.sistema == "modelo") & (mur.grupo == "2024")].iloc[0]
    add(
        "6-res",
        f"cae a {num(m25.resolucion, 3)}, frente a {num(m23.resolucion, 3)} en 2023 y {num(m24.resolucion, 3)} en 2024",
    )
    c21 = dcd[dcd.anio == 2021].set_index("municipality_code")
    add(
        "6-2021",
        f"(BSS {num(c21.loc['54001', 'bss_b2'], 3, True)}), y en Cali, post-epidémica, por un margen mínimo "
        f"({num(c21.loc['76001', 'bss_b2'], 3, True)})",
    )

    # ---------------------------------------------------------------- Table 7.1 and section 7
    re_ = drp[(drp.nivel == "regimen") & (drp.grupo == "epidemica")].iloc[0]
    rp = drp[(drp.nivel == "regimen") & (drp.grupo == "post_epidemica")].iloc[0]
    add(
        "T7-a",
        f"| {re_.victorias_logloss_b2} de {re_.casillas} (por Brier, {re_.victorias_brier_b2} de "
        f"{re_.casillas}); en las "
        f"filas juntas, mejora de log-loss {pct(re_.mejora_logloss_b2_filas, 1, True)}, BSS "
        f"{num(re_.bss_b2_filas, 3, True)} |",
    )
    add(
        "T7-b",
        f"| Pierde {rp.casillas - rp.victorias_logloss_b2} de {rp.casillas} (Barranquilla 2025); en las filas juntas, "
        f"{pct(rp.mejora_logloss_b2_filas, 1, True)} de log-loss y BSS {num(rp.bss_b2_filas, 3, True)} |",
    )
    post = dcp[dcp.regimen == "post_epidemica"]
    n_over = int((post.calibracion_modelo > post.calibracion_b2).sum())
    add(
        "T7-c",
        f"| {n_over} de {len(post)}: su calibración en grande es menor que la de B2 en las cuatro (filas juntas: "
        f"{num(rp.calibracion_modelo_filas, 3, True)} frente a {num(rp.calibracion_b2_filas, 3, True)}) |",
    )
    pdv = dcd[dcd.regimen == "post_epidemica"]
    p20 = pdv[pdv.anio == 2020]
    over20 = p20[p20.calibracion_modelo > p20.calibracion_b2].municipality_code.map(MUNIS).tolist()
    assert len(p20) == 4 and over20 == ["Soledad", "Neiva", "Villavicencio"], (len(p20), over20)
    rpd = drd[(drd.nivel == "regimen") & (drd.grupo == "post_epidemica")].iloc[0]
    add(
        "7-dev",
        f"el modelo ganaba {int(pdv.gana_logloss_b2.sum())} de {len(pdv)} y sobrestimaba más que B2 en tres de las "
        f"cuatro de 2020 (Soledad, Neiva y Villavicencio), aunque no en las filas juntas "
        f"({num(rpd.calibracion_modelo_filas, 3, True)} frente a {num(rpd.calibracion_b2_filas, 3, True)}",
    )
    add("7-post", f"tienen {post.positivas.sum()} semanas positivas en {post.filas.sum()} filas")
    add(
        "7-x8",
        f"el p̂ medio ({num(s13.p_medio_modelo, 2)}) multiplica por ocho la frecuencia observada "
        f"({num(s13.prevalencia, 3)})",
    )
    assert round(s13.p_medio_modelo / s13.prevalencia) == 8
    dw = ev_g["delta_w"]
    add(
        "T7-he",
        f"{sum(d['delta_w'] > 0 for d in dw)} de {len(dw)} (Δw de "
        + ", ".join(f"{'+' if d['delta_w'] > 0 else ''}{int(d['delta_w'])}" for d in dw).rsplit(", ", 1)[0]
        + " y +"
        + str(int(dw[-1]["delta_w"]))
        + ")",
    )

    # ---------------------------------------------------------------- section 8
    hp = H("prueba", "modelo", "B2")
    add(
        "8.1-a",
        f"el BSS promedio de las casillas ({num(hp.bss_casillas, 3, True)}) duplica el de las filas juntas "
        f"({num(hp.bss_filas, 3, True)})",
    )
    add(
        "8.1-b",
        f"(Cali 2025, {num(cell(dcp, '76001', 2025).bss_b2, 3, True)}; Villavicencio 2024, "
        f"{num(cell(dcp, '50001', 2024).bss_b2, 3, True)}; Neiva 2025, "
        f"{num(cell(dcp, '41001', 2025).bss_b2, 3, True)})",
    )
    hd = H("desarrollo", "ablacion", "B2")
    add(
        "8.1-c",
        f"{num(hd.bss_casillas, 3, True)} como promedio de casillas y {num(hd.bss_filas, 3, True)} con las "
        f"filas juntas",
    )

    def r81(label, per, sis, ref):
        h = H(per, sis, ref)
        return row(
            label,
            num(h.bss_filas, 3, True),
            num(h.bss_casillas, 3, True),
            pct(h.mejora_logloss_filas, 1, True),
            pct(h.mejora_logloss_casillas, 1, True),
        )

    add("T8.1-1", r81("Modelo frente a B2, prueba", "prueba", "modelo", "B2"))
    add("T8.1-2", r81("Modelo frente a B2, desarrollo", "desarrollo", "modelo", "B2"))
    add("T8.1-3", r81("Modelo frente a B1, prueba", "prueba", "modelo", "B1"))
    add("T8.1-4", r81("Con clima frente a B2, prueba", "prueba", "ablacion", "B2"))
    add("T8.1-5", r81("Con clima frente a B2, desarrollo", "desarrollo", "ablacion", "B2"))
    nop = int((dcp.positivas == 0).sum()), int((dcd.positivas == 0).sum())
    add("8.2-a", f"{nop[0]} en prueba (Neiva y Cali 2025) y {nop[1]} en desarrollo")
    vb = ape[(ape.periodo == "prueba") & (ape.sistema == "B2") & (ape.grupo != "todos")]
    add("8.2-b", f"(entre {vb.valores_distintos.min()} y {vb.valores_distintos.max()} por municipio en prueba)")
    ab = vb.set_index("municipio")
    add(
        "8.2-c",
        f"{num(ab.loc['Barranquilla', 'ap_orden_de_filas'], 2)} en lugar de "
        f"{num(ab.loc['Barranquilla', 'ap_empates_agrupados'], 2)} "
        f"para B2 en Barranquilla y {num(ab.loc['Cúcuta', 'ap_orden_de_filas'], 2)} en lugar de "
        f"{num(ab.loc['Cúcuta', 'ap_empates_agrupados'], 2)} en Cúcuta",
    )
    mod = {c: ficha["municipios"][c]["estimacion"]["ap"]["modelo"] for c in ORDER}
    assert ab.loc["Cúcuta", "ap_orden_de_filas"] > mod["54001"] > ab.loc["Cúcuta", "ap_empates_agrupados"]
    assert ab.loc["Barranquilla", "ap_empates_agrupados"] > mod["08001"]
    aucs = [z(pp, zz).auc_b2 for pp in ("prueba", "desarrollo") for zz in ("alerta", "epidemia")]
    add("8.4", f"(entre {num(min(aucs), 2)} y {num(max(aucs), 2)})")
    add("8.5", f"las entradas ({ee(mp)} en prueba)")
    b2v = mb2[(mb2.periodo == "prueba") & (mb2.sistema == "B2") & mb2.agrupamiento.str.startswith("un grupo")].iloc[0]
    add(
        "8.7",
        f"la confiabilidad de B2 en prueba es {num(b2m.confiabilidad, 3)} y su resolución, "
        f"{num(b2m.resolucion, 3)}. Si se "
        f"agrupa por sus valores exactos ({b2v.grupos} grupos, residuo 0), son {num(b2v.confiabilidad, 3)} y "
        f"{num(b2v.resolucion, 3)}",
    )
    t05 = stau[(stau.sistema == "modelo") & (stau.tau.round(2) == 0.05)].iloc[0]
    add(
        "8.8",
        f"con τ = 0,05, {num(t05.false_alarms_per_municipality_year, 2)} falsas alarmas por municipio-año frente a "
        f"{num(cd.falsas_por_municipio_anio, 2)} del canal",
    )
    add("8.8b", f"{int(t05.false_alarms)} falsas alarmas en {int(t05.municipality_years)} municipio-años")
    add(
        "8.8c",
        f"cuentan {int(md.falsas_alarmas)} falsas alarmas ({num(md.falsas_por_municipio_anio, 2)} por municipio-año)",
    )
    add(
        "8.2-d",
        f"En Cúcuta eso invertiría la comparación con el modelo ({num(mod['54001'], 2)}); en Barranquilla ampliaría "
        f"la ventaja que B2 ya tiene con empates agrupados ({num(ab.loc['Barranquilla', 'ap_empates_agrupados'], 2)} "
        f"frente a {num(mod['08001'], 2)})",
    )

    # ---------------------------------------------------------------- Table 9.1 and 9
    vb2d, vb2p = (
        dvp[(dvp.periodo == "desarrollo") & (dvp.sistema == "modelo") & (dvp.referencia == "B2")].iloc[0],
        dvp[(dvp.periodo == "prueba") & (dvp.sistema == "modelo") & (dvp.referencia == "B2")].iloc[0],
    )
    add(
        "T9-prev",
        row(
            "Prevalencia",
            pct(cal[(cal.periodo == "desarrollo") & (cal.anio == "todos")].iloc[0].prevalencia, 1),
            pct(cal[(cal.periodo == "prueba") & (cal.anio == "todos")].iloc[0].prevalencia, 1),
        ),
    )
    add(
        "T9-bss",
        row("BSS del modelo frente a B2 (filas juntas)", num(vb2d.bss_filas, 3, True), num(vb2p.bss_filas, 3, True)),
    )
    add(
        "T9-ll",
        row(
            "Mejora de log-loss frente a B2",
            pct(vb2d.mejora_logloss_filas, 1, True),
            pct(vb2p.mejora_logloss_filas, 1, True),
        ),
    )
    add(
        "T9-cas",
        row(
            "Casillas ganadas a B2 (Brier)",
            f"{int(vb2d.ganadas_brier)} de 24 ({pct(vb2d.ganadas_brier / 24)})",
            f"{int(vb2p.ganadas_brier)} de 18 ({pct(vb2p.ganadas_brier / 18)})",
        ),
    )
    add(
        "T9-b1",
        row(
            "BSS frente a B1",
            num(H("desarrollo", "modelo", "B1").bss_filas, 3, True),
            num(H("prueba", "modelo", "B1").bss_filas, 3, True),
        ),
    )
    add(
        "T9-abl",
        row(
            "Versión con clima: BSS frente a B2",
            num(H("desarrollo", "ablacion", "B2").bss_filas, 3, True),
            num(H("prueba", "ablacion", "B2").bss_filas, 3, True),
        ),
    )
    add(
        "T9-ap",
        row(
            "AP, modelo / B2 [X]",
            f"{num(dvz.ap_modelo, 2)} / {num(dvz.ap_b2, 2)}",
            f"{num(tz.ap_modelo, 2)} / {num(tz.ap_b2, 2)}",
        ),
    )
    add(
        "T9-mcc",
        row(
            "MCC con τ = 0,30, modelo / persistencia [X]",
            f"{num(m30d.mcc, 2)} / {num(p30d.mcc, 2)}",
            f"{num(m30.mcc, 2)} / {num(p30.mcc, 2)}",
        ),
    )
    add(
        "T9-det",
        row(
            "Detección, modelo / canal",
            f"{md.inicios_avisados} / {cd.inicios_avisados} de 10",
            f"{mp.inicios_avisados} / {cp.inicios_avisados} de 10",
        ),
    )
    add(
        "T9-fa",
        row(
            "Falsas alarmas por municipio-año, modelo / canal",
            f"{num(md.falsas_por_municipio_anio, 2)} / {num(cd.falsas_por_municipio_anio, 2)}",
            f"{num(mp.falsas_por_municipio_anio, 2)} / {num(cp.falsas_por_municipio_anio, 2)}",
        ),
    )
    add(
        "9-marg",
        f"superó a la segunda por "
        f"{num(sel['paso_1_configuracion_comun']['logistica']['ventaja_sobre_segundo'], 4)} de mejora media",
    )
    add(
        "9-auc",
        f"AUC {num(dvz.auc_modelo, 2)} → {num(tz.auc_modelo, 2)}, con una AUC de la persistencia que también cae, de "
        f"{num(dvz.auc_persistencia, 2)} a {num(tz.auc_persistencia, 2)}",
    )

    # ---------------------------------------------------------------- prose repeats of table values
    def fm_(c):
        return ficha["municipios"][c]

    def devbss(c):
        dv = dcd[dcd.municipality_code == c]
        return 1 - (dv.filas * dv.brier_modelo).sum() / (dv.filas * dv.brier_b2).sum()

    def ev_txt(c):
        return fm_(c)["eventos"]

    n_ = ev_txt("41001")
    add(
        "P-neiva",
        f"(BSS {num(fm_('41001')['estimacion']['bss_vs_b2'], 3, True)}; casillas de 2023 a 2025: "
        f"{num(cell(dcp, '41001', 2023).bss_b2, 2, True)}, {num(cell(dcp, '41001', 2024).bss_b2, 2, True)} y "
        f"{num(cell(dcp, '41001', 2025).bss_b2, 2, True)}). En desarrollo gana en tres de cuatro, con BSS "
        f"{num(devbss('41001'), 3, True)}",
    )
    add(
        "P-neiva-ev",
        f"Avisa {n_['avisados_modelo']} de {n_['inicios_evaluables']} inicios, frente a {n_['avisados_canal']} de "
        f"{n_['inicios_evaluables']} del canal, y {n_['entradas']['avisadas_modelo']} de "
        f"{n_['entradas']['evaluables']} "
        f"entradas, frente a {n_['entradas']['avisadas_canal']} de {n_['entradas']['evaluables']}. Falsas alarmas: "
        f"{n_['falsas_alarmas']['modelo']} frente a {n_['falsas_alarmas']['canal']}",
    )
    nd8 = dep[(dep.tipo == "inicio") & (dep.semana == "2023-08-28")].iloc[0]
    add(
        "P-neiva-dw",
        f"el Δw de +{int(nd8.delta_w)} semanas del inicio del 2023-08-28 ({int(nd8.anticipacion_modelo)} semanas de "
        f"anticipación frente a {int(nd8.anticipacion_canal)} del canal)",
    )
    add(
        "P-villa",
        f"(BSS {num(fm_('50001')['estimacion']['bss_vs_b2'], 3, True)}, gana 2023 y 2024) y el municipio con la mayor "
        f"mejora en desarrollo ({num(devbss('50001'), 3, True)}, tres de cuatro años)",
    )
    assert max(ORDER, key=devbss) == "50001"
    v_ = ev_txt("50001")
    add(
        "P-villa-ev",
        f"({v_['falsas_alarmas']['modelo']} frente a {v_['falsas_alarmas']['canal']}), y no avisa ninguna de sus "
        f"{v_['entradas']['evaluables']} entradas, frente a {v_['entradas']['avisadas_canal']} del canal",
    )
    add(
        "P-cali",
        f"(+{num(fm_('76001')['estimacion']['bss_vs_b2'], 3)}) y la mayor AP "
        f"({num(fm_('76001')['estimacion']['ap']['modelo'], 2)} "
        f"frente a {num(fm_('76001')['estimacion']['ap']['B2'], 2)}), pero en desarrollo el BSS es "
        f"{num(devbss('76001'), 3, True)}",
    )
    assert max(ORDER, key=lambda c: fm_(c)["estimacion"]["bss_vs_b2"]) == "76001"
    s_ = ev_txt("08758")
    add(
        "P-sol-ent",
        f"avisa {s_['entradas']['avisadas_canal']} de {s_['entradas']['evaluables']} frente a "
        f"{s_['entradas']['avisadas_modelo']} de {s_['entradas']['evaluables']} del modelo",
    )
    add(
        "P-sol-dent",
        f"(en desarrollo, {sdc.entradas_avisadas} de {ee(sdc)} frente a {sdm.entradas_avisadas} de {ee(sdm)})",
    )
    add(
        "P-sol-ini",
        f"Avisa {s_['avisados_modelo']} de {s_['inicios_evaluables']}, frente a {s_['avisados_canal']} de "
        f"{s_['inicios_evaluables']} del canal",
    )
    add("P-sol-fa", f"Falsas alarmas: {s_['falsas_alarmas']['modelo']} frente a {s_['falsas_alarmas']['canal']}.")
    sol = em[(em.periodo == "prueba") & (em.code == "08758")].set_index("estrato")
    add(
        "P-sol-estr",
        f"(+{num(sol.loc['dentro', 'mejora_ll_b2'] * 100, 0)} %) se compensa con la pérdida fuera de ella "
        f"({pct(sol.loc['fuera', 'mejora_ll_b2'], 0, True)})",
    )
    add("P-sol-dev", f"En desarrollo, BSS {num(devbss('08758'), 3, True)}, con dos de cuatro años")
    k_ = ev_txt("54001")
    add("P-cuc", f"En prueba el BSS es {num(fm_('54001')['estimacion']['bss_vs_b2'], 3, True)}")
    add("P-cuc-dev", f"En desarrollo fue favorable ({num(devbss('54001'), 3, True)}, tres de cuatro años)")
    add(
        "P-cuc-ev",
        f"({k_['entradas']['avisadas_modelo']} de {k_['entradas']['evaluables']} frente a "
        f"{k_['entradas']['avisadas_canal']} de {k_['entradas']['evaluables']}) y tiene un tercio de sus falsas "
        f"alarmas ({k_['falsas_alarmas']['modelo']} frente a {k_['falsas_alarmas']['canal']})",
    )
    kd = dep[(dep.tipo == "inicio") & (dep.semana == "2024-01-15")].iloc[0]
    add("P-cuc-dw", f"con {int(kd.anticipacion_modelo)} semanas de anticipación (Δw = +{int(kd.delta_w)})")
    add("P-bq-dev", f"En desarrollo fue favorable ({num(devbss('08001'), 3, True)}, tres de cuatro años)")
    b_ = ev_txt("08001")
    add("P-bq-fa", f"Falsas alarmas: {b_['falsas_alarmas']['modelo']} frente a {b_['falsas_alarmas']['canal']}.")
    add(
        "P-5-fa",
        f"(1,00 frente a 1,67), pero porque emite menos alertas ({mp.alertas_juzgables} juzgables frente a "
        f"{cp.alertas_juzgables}), no porque sean más precisas: el {pct(mp.proporcion_falsas_alarmas)} de sus alertas "
        f"juzgables es falsa, frente al {pct(cp.proporcion_falsas_alarmas)} del canal",
    )
    add(
        "P-5-dur",
        f"(mediana de {num(dup.duracion_mediana, 0)} semanas desde la confirmación, frente a "
        f"{num(dcn.duracion_mediana, 0)}) "
        f"y pasa bajo alerta el {pct(dup.fraccion_bajo_alerta)} de las semanas, frente al "
        f"{pct(dcn.fraccion_bajo_alerta)} del canal",
    )
    add("P-5-dev", f"({pct(md.proporcion_falsas_alarmas)} frente a {pct(cd.proporcion_falsas_alarmas)})")
    add(
        "P-5-ent",
        f"{mp.entradas_avisadas} de {ee(mp)} frente a {cp.entradas_avisadas} de {ee(cp)} (en desarrollo, "
        f"{md.entradas_avisadas} de {ee(md)} frente a {cd.entradas_avisadas} de {ee(cd)})",
    )
    add(
        "P-5-tb",
        f"bajo alerta en la semana de {tbp.inicios_bajo_alerta} de los 10 inicios. Pero con el "
        f"{pct(tbp.fraccion_bajo_alerta)} de las semanas de emisión bajo alerta",
    )
    add(
        "P-5-mcn",
        f"En los {mc['discrepantes']} inicios en que discrepan, el modelo gana uno (Soledad 2024-12-23) y el "
        f"canal cinco",
    )
    add("P-7-he", f"| {mp.inicios_avisados} frente a {cp.inicios_avisados} de 10 |")
    add("P-7-he2", f"| {num(mp.falsas_por_municipio_anio, 2)} frente a {num(cp.falsas_por_municipio_anio, 2)} |")
    add(
        "P-7-he3",
        f"la de Neiva duró {ne.loc['2023-07-17', 'semanas_desde_tc']} semanas desde t_c y la de Cúcuta, "
        f"{cu_.loc['2024-01-15', 'semanas_desde_tc']}",
    )
    for yy in ("2019", "2020", "2021", "2022", "2023", "2024", "2025"):
        rs = allr[(allr.nivel == "anio") & (allr.grupo == yy)].iloc[0]
        add(f"P-6-{yy}", f"BSS {num(rs.bss_b2_filas, 3, True)}")
    add(
        "P-6-2019",
        f"(calibración {num(cal[cal.anio == '2019'].iloc[0].calibracion_B2, 3, True)} frente a "
        f"{num(cal[cal.anio == '2019'].iloc[0].calibracion_modelo, 3, True)})",
    )
    add(
        "P-6-2020",
        f"(+{num(cal[cal.anio == '2020'].iloc[0].calibracion_modelo, 3)} frente a "
        f"+{num(cal[cal.anio == '2020'].iloc[0].calibracion_B2, 3)})",
    )
    add("P-6-2020b", f"con solo {int(cal[cal.anio == '2020'].iloc[0].positivas)} semanas positivas")
    add(
        "P-6-2024",
        f"pierde Barranquilla ({num(cell(dcp, '08001', 2024).bss_b2, 3, True)}). La calibración en grande del modelo "
        f"es casi exacta ({num(cal[cal.anio == '2024'].iloc[0].calibracion_modelo, 3, True)})",
    )
    add("P-6-2022", f"Epidémico en Barranquilla ({cell(dcd, '08001', 2022).positivas} de 52 semanas positivas)")
    add("P-9-a", f"(BSS {num(vb2d.bss_filas, 3, True)} → {num(vb2p.bss_filas, 3, True)})")
    add(
        "P-9-b",
        f"(+{num(H('desarrollo', 'ablacion', 'B2').bss_filas, 3)} → "
        f"+{num(H('prueba', 'ablacion', 'B2').bss_filas, 3)})",
    )
    add("P-9-c", f"(AP {num(dvz.ap_modelo, 2)} → {num(tz.ap_modelo, 2)};")
    add("P-9-d", f"({md.inicios_avisados} de 10 → {mp.inicios_avisados} de 10)")
    add(
        "P-10-a",
        f"(BSS {num(est['bss_vs_b2'], 3, True)}; {b2c['casillas_ganadas_brier']} de 18 casillas); frente a B1, "
        f"«mejora consistente» (BSS {num(est['bss_vs_b1'], 3, True)}; {b1c['casillas_ganadas_brier']} de 18)",
    )
    add(
        "P-10-b",
        f"(confiabilidad {num(mt.confiabilidad, 3)} frente a {num(b2m.confiabilidad, 3)} con 10 intervalos; "
        f"calibración en "
        f"grande más cerca de 0 los tres años) y ordena mejor las semanas (AP {num(tz.ap_modelo, 2)} frente a "
        f"{num(tz.ap_b2, 2)})",
    )
    assert all(
        abs(cp_.loc[yy, "calibracion_modelo"]) < abs(cp_.loc[yy, "calibracion_B2"]) for yy in ("2023", "2024", "2025")
    )
    add(
        "P-10-c",
        f"recupera el {pct(fu.sensibilidad)} de las entradas y anticipa el {pct(de.especificidad)} de las salidas",
    )
    add(
        "P-10-d",
        f"durante los episodios ({pct(fp_.loc['durante', 'porcentaje_mejora_neta'] / 100, 0, True)}) y en las semanas "
        f"tranquilas ({pct(fp_.loc['resto', 'porcentaje_mejora_neta'] / 100, 0, True)}), y se pierde en las "
        f"1–13 semanas "
        f"posteriores a un episodio ({pct(fp_.loc['después', 'porcentaje_mejora_neta'] / 100, 0, True)})",
    )
    add(
        "P-10-e",
        f"bajo alerta en {tbp.inicios_bajo_alerta} de 10 inicios no se distingue de su tasa base de semanas "
        f"bajo alerta "
        f"(unos {num(tbp.esperado_por_tasa_base, 0)} de 10)",
    )
    add(
        "P-10-f",
        f"({pct(la.loc['[0.00, 0.05)', 'p_medio_modelo'])} predicho, "
        f"{pct(la.loc['[0.00, 0.05)', 'frecuencia_observada'])} observado)",
    )
    add("P-4-mod", f"(0,51 predicho frente a {pct(s13.prevalencia, 1)} observado en prueba)")
    add(
        "P-4-mod2",
        f"entradas a la zona epidémica ({pct(fu.sensibilidad)}) y una de cada seis o siete salidas "
        f"({pct(de.especificidad)})",
    )
    assert 6 < 1 / de.especificidad < 7 and 4 < 1 / fu.sensibilidad < 5
    add("P-4-no", f"el canal avisa {cp.inicios_avisados} de 10 y el modelo {mp.inicios_avisados} de 10")
    add("P-11", f"Tres años, {len(dcp)} casillas correlacionadas y {mp.inicios_evaluables} inicios evaluables")
    return C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analisis", type=Path, default=Path("docs/forecasting/analisis_resultados.md"))
    ap.add_argument("--evidencia", type=Path, default=Path("reports/forecasting/evidence"))
    a = ap.parse_args(argv)
    text = a.analisis.read_text(encoding="utf-8")
    checks = build_checks(Ev(a.evidencia))
    bad = [(cid, t) for cid, t in checks if t not in text]
    for cid, t in bad:
        print(f"FALTA {cid}: {t}")
    print(f"{len(checks) - len(bad)} de {len(checks)} comprobaciones encontradas en el documento.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

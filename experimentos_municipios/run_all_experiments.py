import subprocess
import sys
from pathlib import Path
import pandas as pd

MUNICIPIOS = {
    "54001": "Cúcuta",
    "08001": "Barranquilla",
    "41001": "Neiva",
    "50001": "Villavicencio",
    "76001": "Cali",
    "08758": "Soledad",
}

TARGET_TYPES = ["exceso_simple", "racha_brote"]
MODELS = ["baseline", "lightgbm", "xgboost", "random_forest", "catboost"]

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent if script_dir.name == "experimentos_municipios" else script_dir
    while not (repo_root / ".git").exists() and repo_root.parent != repo_root:
        repo_root = repo_root.parent

    python_executable = sys.executable
    resultados = []

    print("\n" + "="*110)
    print("INICIANDO SUITE COMPLETA DE EXPERIMENTOS TERRITORIALES (Métricas + Matrices + Variables)")
    print("="*110)

    for code, name in MUNICIPIOS.items():
        for target in TARGET_TYPES:
            for model in MODELS:
                if model == "baseline":
                    cmd = [python_executable, str(script_dir / "train_baseline.py"), "--municipality_code", code, "--target_type", target, "--horizonte", "4"]
                else:
                    cmd = [python_executable, str(script_dir / "train_ensembles.py"), "--municipality_code", code, "--model", model, "--target_type", target, "--horizonte", "4"]

                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"  [ERROR] Falló {model.upper()} en {name} ({target}):")
                    print(result.stderr[-300:])
                else:
                    summary_line = next((line for line in result.stdout.splitlines() if line.startswith("METRICAS_SUMMARY|")), None)
                    features_line = next((line for line in result.stdout.splitlines() if line.startswith("TOP_FEATURES|")), None)

                    if summary_line:
                        _, mean_auc_pr, mean_f1, prec, rec, tn, fp, fn, tp = summary_line.split("|")
                        top_feats = features_line.split("|")[1] if features_line else "N/A"
                        
                        print(f"[{name[:12]:<12}] {model.upper():<13} | {target[:13]:<13} | "
                              f"F1: {mean_f1} | Prec: {prec} | Rec: {rec} | "
                              f"CM: [[{tn}, {fp}], [{fn}, {tp}]]")

                        resultados.append({
                            "municipio_cod": code,
                            "municipio_nom": name,
                            "target_type": target,
                            "modelo": model,
                            "mean_auc_pr": float(mean_auc_pr),
                            "mean_f1_score": float(mean_f1),
                            "oot_precision": float(prec),
                            "oot_recall": float(rec),
                            "tn": int(tn),
                            "fp": int(fp),
                            "fn": int(fn),
                            "tp": int(tp),
                            "top_variables": top_feats
                        })

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(resultados)
    output_path = reports_dir / "resumen_experimentos_completo.csv"
    df_res.to_csv(output_path, index=False)

    print("\n" + "="*110)
    print(f"EJECUCIÓN COMPLETADA. Resumen Maestro guardado en: {output_path}")
    print("="*110)
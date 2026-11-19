"""Contraste de Deep Learning secuencial (GRU y CNN-1D) en PyTorch sobre GPU.

Usa build_gru_examples, yearly_trials, B2 y métricas oficiales del contrato (2019-2022).
Evalúa variantes sin_clima (5 canales) y con_clima (8 canales) en igualdad de condiciones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.forecasting.baselines import b2_predict
from src.forecasting.config import load_config
from src.forecasting.dataset import ZONES, build_gru_examples, load_channel_product
from src.forecasting.guard import RecipeAccess
from src.forecasting.metrics import brier, improvement, log_loss
from src.forecasting.splits import yearly_trials


class CompactGRU(nn.Module):
    def __init__(self, n_seq: int, n_static: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_seq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + n_static, 1)

    def forward(self, x_seq: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x_seq)
        feat = torch.cat([self.drop(h_n[-1]), x_stat], dim=1)
        return self.head(feat).squeeze(-1)


class CompactCNN1D(nn.Module):
    def __init__(self, n_seq: int, n_static: int, channels: int, kernel_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(n_seq, channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if num_layers == 2:
            layers.extend([
                nn.Conv1d(channels, channels, kernel_size=kernel_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(channels + n_static, 1)

    def forward(self, x_seq: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        h = self.conv(x_seq.transpose(1, 2)).mean(dim=-1)
        feat = torch.cat([h, x_stat], dim=1)
        return self.head(feat).squeeze(-1)


def sample_candidates(family: str, n: int, seed: int) -> list[tuple[str, dict]]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lr = float(10 ** rng.uniform(np.log10(3e-4), np.log10(1.5e-2)))
        wd = float(10 ** rng.uniform(np.log10(1e-4), np.log10(5e-1)))
        dropout = float(rng.uniform(0.0, 0.45))
        epochs = int(rng.integers(40, 181))
        batch_size = int(rng.choice([64, 128, 256]))
        num_layers = int(rng.choice([1, 2]))
        if family == "gru":
            params = {
                "hidden_size": int(rng.choice([8, 12, 16, 24, 32, 48])),
                "num_layers": num_layers,
                "dropout": round(dropout, 4),
                "lr": round(lr, 6),
                "weight_decay": round(wd, 6),
                "epochs": epochs,
                "batch_size": batch_size,
            }
        else:
            params = {
                "channels": int(rng.choice([8, 12, 16, 24, 32, 48])),
                "kernel_size": int(rng.choice([2, 3, 4])),
                "num_layers": num_layers,
                "dropout": round(dropout, 4),
                "lr": round(lr, 6),
                "weight_decay": round(wd, 6),
                "epochs": epochs,
                "batch_size": batch_size,
            }
        out.append((f"{family}_{i:02d}", params))
    return out


def standardize_fold(
    seq_tr: np.ndarray,
    seq_ev: np.ndarray,
    stat_tr: np.ndarray,
    stat_ev: np.ndarray,
    cont_seq_idx: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_tr = seq_tr.copy()
    s_ev = seq_ev.copy()
    for c in cont_seq_idx:
        mu = float(s_tr[:, :, c].mean())
        sd = float(s_tr[:, :, c].std())
        sd = 1.0 if sd < 1e-8 else sd
        s_tr[:, :, c] = (s_tr[:, :, c] - mu) / sd
        s_ev[:, :, c] = (s_ev[:, :, c] - mu) / sd

    st_tr = stat_tr.copy()
    st_ev = stat_ev.copy()
    # Escalar cambio_umbral (índice 0); sem_sin y sem_cos ya están en [-1, 1]
    mu0 = float(st_tr[:, 0].mean())
    sd0 = float(st_tr[:, 0].std())
    sd0 = 1.0 if sd0 < 1e-8 else sd0
    st_tr[:, 0] = (st_tr[:, 0] - mu0) / sd0
    st_ev[:, 0] = (st_ev[:, 0] - mu0) / sd0
    return s_tr, s_ev, st_tr, st_ev


def fit_and_predict(
    family: str,
    params: dict,
    seq_tr: np.ndarray,
    stat_tr: np.ndarray,
    y_tr: np.ndarray,
    seq_ev: np.ndarray,
    stat_ev: np.ndarray,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    n_seq, n_static = seq_tr.shape[2], stat_tr.shape[1]
    if family == "gru":
        model = CompactGRU(n_seq, n_static, params["hidden_size"], params["num_layers"], params["dropout"]).to(device)
    else:
        model = CompactCNN1D(
            n_seq, n_static, params["channels"], params["kernel_size"], params["num_layers"], params["dropout"]
        ).to(device)

    x_seq_t = torch.tensor(seq_tr, dtype=torch.float32, device=device)
    x_stat_t = torch.tensor(stat_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    n_samples = len(y_tr)
    bs = params["batch_size"]
    model.train()
    for _ in range(params["epochs"]):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, bs):
            idx = perm[start : start + bs]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_seq_t[idx], x_stat_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        ev_seq_t = torch.tensor(seq_ev, dtype=torch.float32, device=device)
        ev_stat_t = torch.tensor(stat_ev, dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(ev_seq_t, ev_stat_t)).cpu().numpy().astype(float)
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--familias", nargs="+", default=["gru", "cnn1d"], choices=["gru", "cnn1d"])
    parser.add_argument("--variantes", nargs="+", default=["sin_clima", "con_clima"], choices=["sin_clima", "con_clima"])
    parser.add_argument("--n-candidatos", type=int, default=25)
    parser.add_argument("--semilla", type=int, default=20261112)
    parser.add_argument("--salida", type=Path, default=Path("data/processed/forecasting/oscar_dl_secuencial"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo activo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    cfg = load_config()
    access = RecipeAccess.from_config(cfg)
    prod = load_channel_product(cfg.datos.canal, expected_md5=cfg.datos.version_dvc_canal)
    gru_data = build_gru_examples(prod, h=cfg.horizonte.h, canal=cfg.canal, features=cfg.features)

    keys = gru_data.keys.reset_index(drop=True)
    keys["zona_t"] = [ZONES[i] for i in np.argmax(gru_data.sequence[:, -1, 2:5], axis=1)]
    key_to_idx = {(str(r.municipality_code), r.t): i for i, r in enumerate(keys.itertuples(index=False))}
    dev_years = range(cfg.validacion.desarrollo[0], cfg.validacion.desarrollo[1] + 1)
    trials = yearly_trials(keys, dev_years, first_train_year=cfg.validacion.primer_anio_entrenamiento)

    # Precalcular B2 y los índices exactos por pliegue (24 casillas)
    print(f"Secuencias listas: {gru_data.sequence.shape} | Pliegues: {len(trials)}", flush=True)
    fold_cache = []
    for tr in trials:
        tr_idx = keys.index.get_indexer(tr.train)
        ev_idx = keys.index.get_indexer(tr.evaluate)
        train_df, eval_df = keys.loc[tr.train], keys.loc[tr.evaluate]
        y_tr = train_df["y"].to_numpy(dtype=np.float32)
        y_ev = eval_df["y"].to_numpy(dtype=int)
        p_b2 = b2_predict(train_df, eval_df, access=access)
        ll_b2 = log_loss(y_ev, p_b2, eval_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss)
        fold_cache.append((tr, tr_idx, ev_idx, y_tr, y_ev, ll_b2))

    rows = []
    t_global = time.perf_counter()

    for variante in args.variantes:
        seq_cols = [0, 1, 2, 3, 4] if variante == "sin_clima" else [0, 1, 2, 3, 4, 5, 6, 7]
        cont_idx = [0, 1] if variante == "sin_clima" else [0, 1, 5, 6, 7]

        # Pre-escalar los 24 pliegues para esta variante
        scaled_folds = []
        for tr, tr_idx, ev_idx, y_tr, y_ev, ll_b2 in fold_cache:
            s_tr, s_ev, st_tr, st_ev = standardize_fold(
                gru_data.sequence[tr_idx][:, :, seq_cols],
                gru_data.sequence[ev_idx][:, :, seq_cols],
                gru_data.static[tr_idx],
                gru_data.static[ev_idx],
                cont_idx,
            )
            scaled_folds.append((tr, s_tr, s_ev, st_tr, st_ev, y_tr, y_ev, ll_b2))

        for family in args.familias:
            cands = sample_candidates(family, args.n_candidatos, args.semilla)
            for idx_c, (cand_name, params) in enumerate(cands, start=1):
                t0 = time.perf_counter()
                fold_imps = []
                for tr, s_tr, s_ev, st_tr, st_ev, y_tr, y_ev, ll_b2 in scaled_folds:
                    probs = fit_and_predict(family, params, s_tr, st_tr, y_tr, s_ev, st_ev, device, args.semilla)
                    ll = log_loss(y_ev, probs, keys.loc[tr.evaluate, "target_week"], access=access, clip=cfg.metricas.recorte_log_loss)
                    br = brier(y_ev, probs, keys.loc[tr.evaluate, "target_week"], access=access)
                    imp = improvement(ll, ll_b2)
                    fold_imps.append(imp)
                    rows.append({
                        "familia": family,
                        "variante": variante,
                        "candidato": cand_name,
                        "parametros": json.dumps(params, sort_keys=True),
                        "municipality_code": str(tr.municipality_code),
                        "year": int(tr.year),
                        "logloss": float(ll),
                        "brier": float(br),
                        "logloss_b2": float(ll_b2),
                        "mejora_b2": float(imp),
                        "segundos": float(time.perf_counter() - t0),
                    })
                dt = time.perf_counter() - t0
                if True:
                    print(f"[{variante:9s} | {family:5s}] {idx_c:2d}/{len(cands)} {cand_name} -> mejora_b2={np.mean(fold_imps):+.4f} ({dt:.1f}s)", flush=True)

    args.salida.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_path = args.salida / "tabla_unica_desarrollo.parquet"
    out_df.to_parquet(out_path, index=False)

    res = (
        out_df.groupby(["variante", "familia", "candidato", "parametros"], as_index=False)
        .agg(
            mejora_b2_media=("mejora_b2", "mean"),
            logloss_medio=("logloss", "mean"),
            brier_medio=("brier", "mean"),
            casillas_ganadas_b2=("mejora_b2", lambda s: int((s > 0).sum())),
        )
        .sort_values("mejora_b2_media", ascending=False)
    )
    mejores = res.groupby(["variante", "familia"], as_index=False).first().sort_values("mejora_b2_media", ascending=False)
    print(f"\n=== RESUMEN FINAL ({time.perf_counter() - t_global:.1f}s) | Guardado en {out_path} ===")
    print(mejores[["variante", "familia", "candidato", "mejora_b2_media", "logloss_medio", "brier_medio", "casillas_ganadas_b2"]].to_string(index=False))
    print("\n=== MEJORES HIPERPARÁMETROS ===")
    for _, r in mejores.iterrows():
        print(f"{r['variante']} | {r['familia']} ({r['candidato']}): {r['parametros']}")


if __name__ == "__main__":
    main()

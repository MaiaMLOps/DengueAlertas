"""Sequential deep-learning contrast (GRU and CNN-1D, PyTorch) in development, 2019–2022 (análisis complementario).

Uses build_gru_examples, splits.yearly_trials, the B2 baseline and the contract metrics on the 24 development cells.
Evaluates the sin_clima variant (5 channels) and con_clima (8 channels: plus the three weekly climate series) on the
same cells. The space, seed and output folder come from the sequential run of configs/contraste_familias.yaml.

Selection: the table holds every candidate in every cell; scripts/build_thesis_report.py keeps, per family and
variant, the candidate with the largest mean improvement over B2 on these same 24 cells (in-sample; see the README
of reports/tesis_contraste_modelos/).

Determinism: torch.manual_seed fixes the initialisation and the batch order, but results on GPU (CUDA) and CPU are
not bit-identical, and some CUDA kernels are not deterministic between runs.

Run from the repository root (optional dependencies: requirements/contraste.txt):
    .venv/Scripts/python.exe -B scripts/run_seq_dl.py --config configs/contraste_familias.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecasting.baselines import b2_predict  # noqa: E402
from src.forecasting.config import (  # noqa: E402
    CONTRAST_CONFIG_PATH,
    SequentialSpace,
    contrast_base_config,
    load_contrast_config,
    resolve_path,
)
from src.forecasting.dataset import ZONES, build_gru_examples, load_channel_product  # noqa: E402
from src.forecasting.guard import RecipeAccess  # noqa: E402
from src.forecasting.metrics import brier, improvement, log_loss  # noqa: E402
from src.forecasting.splits import yearly_trials  # noqa: E402

# Channels of the sequence (build_gru_examples): 0-1 continuous series of the case counts, 2-4 zone indicators,
# 5-7 climate series. Standardised channels are the continuous ones of each variant.
SEQUENCE_CHANNELS = {"sin_clima": [0, 1, 2, 3, 4], "con_clima": [0, 1, 2, 3, 4, 5, 6, 7]}
CONTINUOUS_CHANNELS = {"sin_clima": [0, 1], "con_clima": [0, 1, 5, 6, 7]}


def import_torch():
    """torch is an optional dependency of the contrast; it is imported only when a network is trained."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is needed for the sequential contrast; install requirements/contraste.txt "
                                  "(see reports/tesis_contraste_modelos/README.md).") from exc
    return torch


def build_network(family: str, params: dict, n_seq: int, n_static: int):
    """Compact GRU (last hidden state) or CNN-1D (time-averaged convolutions), joined with the static inputs."""
    torch = import_torch()
    nn = torch.nn

    class CompactGRU(nn.Module):
        def __init__(self, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.gru = nn.GRU(input_size=n_seq, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size + n_static, 1)

        def forward(self, x_seq, x_stat):
            _, h_n = self.gru(x_seq)
            return self.head(torch.cat([self.drop(h_n[-1]), x_stat], dim=1)).squeeze(-1)

    class CompactCNN1D(nn.Module):
        def __init__(self, channels: int, kernel_size: int, num_layers: int, dropout: float):
            super().__init__()
            layers = [nn.Conv1d(n_seq, channels, kernel_size=kernel_size), nn.ReLU(), nn.Dropout(dropout)]
            if num_layers == 2:
                layers += [nn.Conv1d(channels, channels, kernel_size=kernel_size), nn.ReLU(), nn.Dropout(dropout)]
            self.conv = nn.Sequential(*layers)
            self.head = nn.Linear(channels + n_static, 1)

        def forward(self, x_seq, x_stat):
            h = self.conv(x_seq.transpose(1, 2)).mean(dim=-1)
            return self.head(torch.cat([h, x_stat], dim=1)).squeeze(-1)

    if family == "gru":
        return CompactGRU(params["hidden_size"], params["num_layers"], params["dropout"])
    return CompactCNN1D(params["channels"], params["kernel_size"], params["num_layers"], params["dropout"])


def sample_candidates(family: str, space: SequentialSpace, n: int, seed: int) -> list[tuple[str, dict]]:
    """n candidates from numpy default_rng(seed). The draw order is fixed (it defines the candidates of a seed):
    lr, weight_decay, dropout, epochs, batch_size, num_layers, then hidden_size (gru) or channels, kernel_size."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        lr = float(10 ** rng.uniform(np.log10(space.lr[0]), np.log10(space.lr[1])))
        wd = float(10 ** rng.uniform(np.log10(space.weight_decay[0]), np.log10(space.weight_decay[1])))
        dropout = float(rng.uniform(space.dropout[0], space.dropout[1]))
        epochs = int(rng.integers(space.epochs[0], space.epochs[1] + 1))
        batch_size = int(rng.choice(list(space.batch_size)))
        num_layers = int(rng.choice(list(space.num_layers)))
        common = {"num_layers": num_layers, "dropout": round(dropout, 4), "lr": round(lr, 6),
                  "weight_decay": round(wd, 6), "epochs": epochs, "batch_size": batch_size}
        if family == "gru":
            params = {"hidden_size": int(rng.choice(list(space.gru_hidden_size))), **common}
        else:
            params = {"channels": int(rng.choice(list(space.cnn1d_channels))),
                      "kernel_size": int(rng.choice(list(space.cnn1d_kernel_size))), **common}
        out.append((f"{family}_{i:02d}", params))
    return out


def standardize_fold(seq_tr: np.ndarray, seq_ev: np.ndarray, stat_tr: np.ndarray, stat_ev: np.ndarray,
                     cont_seq_idx: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardise the continuous channels and the first static input with training statistics only."""
    s_tr, s_ev = seq_tr.copy(), seq_ev.copy()
    for c in cont_seq_idx:
        mu = float(s_tr[:, :, c].mean())
        sd = float(s_tr[:, :, c].std())
        sd = 1.0 if sd < 1e-8 else sd
        s_tr[:, :, c] = (s_tr[:, :, c] - mu) / sd
        s_ev[:, :, c] = (s_ev[:, :, c] - mu) / sd
    st_tr, st_ev = stat_tr.copy(), stat_ev.copy()
    # Static input 0 (cambio_umbral) is standardised; the seasonal sine and cosine already lie in [-1, 1].
    mu0 = float(st_tr[:, 0].mean())
    sd0 = float(st_tr[:, 0].std())
    sd0 = 1.0 if sd0 < 1e-8 else sd0
    st_tr[:, 0] = (st_tr[:, 0] - mu0) / sd0
    st_ev[:, 0] = (st_ev[:, 0] - mu0) / sd0
    return s_tr, s_ev, st_tr, st_ev


def fit_and_predict(family: str, params: dict, seq_tr: np.ndarray, stat_tr: np.ndarray, y_tr: np.ndarray,
                    seq_ev: np.ndarray, stat_ev: np.ndarray, device, seed: int) -> np.ndarray:
    """Train with AdamW and BCE (no class weights, no early stopping) and return p(y = 1) of the evaluated rows."""
    torch = import_torch()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_network(family, params, seq_tr.shape[2], stat_tr.shape[1]).to(device)
    x_seq_t = torch.tensor(seq_tr, dtype=torch.float32, device=device)
    x_stat_t = torch.tensor(stat_tr, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()
    n_samples, bs = len(y_tr), params["batch_size"]
    model.train()
    for _ in range(params["epochs"]):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, bs):
            idx = perm[start:start + bs]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_seq_t[idx], x_stat_t[idx]), y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        ev_seq_t = torch.tensor(seq_ev, dtype=torch.float32, device=device)
        ev_stat_t = torch.tensor(stat_ev, dtype=torch.float32, device=device)
        return torch.sigmoid(model(ev_seq_t, ev_stat_t)).cpu().numpy().astype(float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential deep-learning contrast (development only).")
    parser.add_argument("--config", type=Path, default=CONTRAST_CONFIG_PATH,
                        help="Family contrast configuration (configs/contraste_familias.yaml).")
    parser.add_argument("--familias", nargs="+", choices=["gru", "cnn1d"], default=None,
                        help="Subset of the configured families (default: all).")
    parser.add_argument("--variantes", nargs="+", choices=["sin_clima", "con_clima"], default=None,
                        help="Subset of the configured variants (default: all).")
    parser.add_argument("--salida", type=Path, default=None, help="Output folder (default: the run's carpeta).")
    args = parser.parse_args(argv)

    contrast = load_contrast_config(args.config)
    run_name, search = contrast.secuencial
    cfg = contrast_base_config(contrast)
    families = args.familias or list(search.familias)
    variants = args.variantes or list(search.variantes)
    output_dir = args.salida or resolve_path(contrast.corridas[run_name].carpeta)
    out_path = output_dir / "tabla_unica_desarrollo.parquet"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists; outputs are never overwritten.")

    torch = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    access = RecipeAccess.from_config(cfg)
    prod = load_channel_product(resolve_path(cfg.datos.canal), expected_md5=cfg.datos.version_dvc_canal)
    gru_data = build_gru_examples(prod, h=cfg.horizonte.h, canal=cfg.canal, features=cfg.features)
    keys = gru_data.keys.reset_index(drop=True)
    keys["zona_t"] = [ZONES[i] for i in np.argmax(gru_data.sequence[:, -1, 2:5], axis=1)]
    dev_years = range(cfg.validacion.desarrollo[0], cfg.validacion.desarrollo[1] + 1)
    trials = yearly_trials(keys, dev_years, first_train_year=cfg.validacion.primer_anio_entrenamiento)
    print(f"Sequences: {gru_data.sequence.shape} | cells: {len(trials)}", flush=True)

    # B2 and the row indices of each of the 24 cells, shared by every candidate (paired comparison).
    fold_cache = []
    for tr in trials:
        tr_idx, ev_idx = keys.index.get_indexer(tr.train), keys.index.get_indexer(tr.evaluate)
        train_df, eval_df = keys.loc[tr.train], keys.loc[tr.evaluate]
        y_ev = eval_df["y"].to_numpy(dtype=int)
        p_b2 = b2_predict(train_df, eval_df, access=access)
        ll_b2 = log_loss(y_ev, p_b2, eval_df["target_week"], access=access, clip=cfg.metricas.recorte_log_loss)
        fold_cache.append((tr, tr_idx, ev_idx, train_df["y"].to_numpy(dtype=np.float32), y_ev, ll_b2))

    rows = []
    t_global = time.perf_counter()
    for variante in variants:
        seq_cols, cont_idx = SEQUENCE_CHANNELS[variante], CONTINUOUS_CHANNELS[variante]
        scaled_folds = []
        for tr, tr_idx, ev_idx, y_tr, y_ev, ll_b2 in fold_cache:
            s_tr, s_ev, st_tr, st_ev = standardize_fold(gru_data.sequence[tr_idx][:, :, seq_cols],
                                                        gru_data.sequence[ev_idx][:, :, seq_cols],
                                                        gru_data.static[tr_idx], gru_data.static[ev_idx], cont_idx)
            scaled_folds.append((tr, s_tr, s_ev, st_tr, st_ev, y_tr, y_ev, ll_b2))
        for family in families:
            candidates = sample_candidates(family, search.espacio, search.n_candidatos, search.semilla)
            for position, (cand_name, params) in enumerate(candidates, start=1):
                t0 = time.perf_counter()
                gains = []
                for tr, s_tr, s_ev, st_tr, st_ev, y_tr, y_ev, ll_b2 in scaled_folds:
                    probs = fit_and_predict(family, params, s_tr, st_tr, y_tr, s_ev, st_ev, device, search.semilla)
                    weeks = keys.loc[tr.evaluate, "target_week"]
                    ll = log_loss(y_ev, probs, weeks, access=access, clip=cfg.metricas.recorte_log_loss)
                    br = brier(y_ev, probs, weeks, access=access)
                    gain = improvement(ll, ll_b2)
                    gains.append(gain)
                    rows.append({"familia": family, "variante": variante, "candidato": cand_name,
                                 "parametros": json.dumps(params, sort_keys=True),
                                 "municipality_code": str(tr.municipality_code), "year": int(tr.year),
                                 "logloss": float(ll), "brier": float(br), "logloss_b2": float(ll_b2),
                                 "mejora_b2": float(gain), "segundos": float(time.perf_counter() - t0)})
                print(f"[{variante:9s} | {family:5s}] {position:2d}/{len(candidates)} {cand_name} -> "
                      f"mejora_b2={np.mean(gains):+.4f} ({time.perf_counter() - t0:.1f}s)", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    summary = (out_df.groupby(["variante", "familia", "candidato", "parametros"], as_index=False)
               .agg(mejora_b2_media=("mejora_b2", "mean"), logloss_medio=("logloss", "mean"),
                    brier_medio=("brier", "mean"), casillas_ganadas_b2=("mejora_b2", lambda s: int((s > 0).sum())))
               .sort_values("mejora_b2_media", ascending=False))
    best = summary.groupby(["variante", "familia"], as_index=False).first().sort_values("mejora_b2_media",
                                                                                        ascending=False)
    print(f"\n=== Summary ({time.perf_counter() - t_global:.1f}s) | written {out_path} ===")
    print(best[["variante", "familia", "candidato", "mejora_b2_media", "logloss_medio", "brier_medio",
                "casillas_ganadas_b2"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

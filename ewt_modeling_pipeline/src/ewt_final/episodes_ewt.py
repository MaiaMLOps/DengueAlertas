"""Episodios y variable respuesta pre-onset de la Parte III de v0.4."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Episode:
    municipality_code: str
    episode_id: int
    start_week: pd.Timestamp
    end_week: pd.Timestamp
    duration_weeks: int

    def to_dict(self) -> dict:
        item = asdict(self)
        item["start_week"] = self.start_week.date().isoformat()
        item["end_week"] = self.end_week.date().isoformat()
        return item


def _weekly_distance(left: pd.Timestamp, right: pd.Timestamp) -> int:
    days = (right - left).days
    if days % 7:
        raise ValueError("La serie debe usar una grilla semanal regular")
    return days // 7


def episodes_from_flags(dates: Iterable, flags: Iterable, municipality_code: str,
                        min_len: int = 3, merge_gap: int = 2) -> list[Episode]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    flags = pd.Series(list(flags), dtype="boolean")
    if len(dates) != len(flags):
        raise ValueError("dates y flags deben tener igual longitud")
    if len(dates) and (dates.duplicated().any() or not dates.is_monotonic_increasing):
        raise ValueError("Las fechas deben ser únicas y ordenadas")
    true_dates = list(dates[flags.fillna(False).to_numpy(dtype=bool)])
    if not true_dates:
        return []

    runs: list[list[pd.Timestamp]] = []
    for date in true_dates:
        if not runs or _weekly_distance(runs[-1][-1], date) != 1:
            runs.append([date, date])
        else:
            runs[-1][-1] = date

    merged: list[list[pd.Timestamp]] = []
    for start, end in runs:
        if merged:
            gap = _weekly_distance(merged[-1][1], start) - 1
            if gap <= merge_gap:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    output: list[Episode] = []
    for start, end in merged:
        duration = _weekly_distance(start, end) + 1
        if duration >= min_len:
            output.append(Episode(str(municipality_code).zfill(5), len(output) + 1, start, end, duration))
    return output


def label_pre_onset(dates: Iterable, channel_known: Iterable, episodes: list[Episode],
                    horizon: int, refractory: int = 8) -> pd.Series:
    """Crea 1 antes del inicio y NaN en brote/refractario.

    La prioridad documentada se aplica explícitamente al final:
    brote/refractario (NaN) > ventana previa (1) > resto conocido (0).
    """
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    known = np.asarray(list(channel_known), dtype=bool)
    if len(dates) != len(known) or horizon < 1 or refractory < 0:
        raise ValueError("Argumentos de etiquetado no válidos")
    index = {date: position for position, date in enumerate(dates)}
    values = np.where(known, 0.0, np.nan)

    # Ventanas positivas primero.
    for episode in episodes:
        start_pos = index.get(episode.start_week)
        if start_pos is None:
            continue
        left = max(0, start_pos - horizon)
        mask = known[left:start_pos]
        values[left:start_pos][mask] = 1.0

    # Brote y refractario enmascaran cualquier ventana que se solape.
    for episode in episodes:
        start_pos = index.get(episode.start_week)
        end_pos = index.get(episode.end_week)
        if start_pos is None or end_pos is None:
            continue
        values[start_pos:min(len(values), end_pos + refractory + 1)] = np.nan
    return pd.Series(values, index=dates, name=f"y_pre_onset_h{horizon}")


def apply_episode_labels(frame: pd.DataFrame, horizons: Iterable[int], min_len: int = 3,
                         merge_gap: int = 2, refractory: int = 8,
                         flag_col: str = "excess_flag_v04",
                         known_col: str = "ch_mu_nb") -> tuple[pd.DataFrame, pd.DataFrame]:
    """flag_col: bandera semanal de exceso; known_col: columna del canal (nula = sin canal).

    NB-GLM: flag_col="excess_flag_v04", known_col="ch_mu_nb" (valores por defecto).
    EWT:    flag_col="excess_flag_ewt", known_col="ch_ewt".
    """
    required = {"municipality_code", "week_start", flag_col, known_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    result = frame.sort_values(["municipality_code", "week_start"]).copy()
    result["episode_active"] = False
    records: list[dict] = []
    for code, group in result.groupby("municipality_code", sort=False):
        eps = episodes_from_flags(group["week_start"], group[flag_col], code, min_len, merge_gap)
        records.extend(ep.to_dict() for ep in eps)
        known = group[known_col].notna()
        for horizon in horizons:
            labels = label_pre_onset(group["week_start"], known, eps, int(horizon), refractory)
            result.loc[group.index, f"y_pre_onset_h{int(horizon)}"] = labels.to_numpy()
        for ep in eps:
            mask = group["week_start"].between(ep.start_week, ep.end_week)
            result.loc[group.index[mask], "episode_active"] = True
    result["episode_active"] = result["episode_active"].fillna(False).astype(bool)
    columns = ["municipality_code", "episode_id", "start_week", "end_week", "duration_weeks"]
    episodes_df = pd.DataFrame.from_records(records, columns=columns)
    return result, episodes_df

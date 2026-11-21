from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ewt_final.ewt_channel import EwtConfig, build_ewt_channel, ewt
from ewt_final.episodes_ewt import episodes_from_flags, label_pre_onset
from ewt_final.features_ewt import ALL_FEATURES, build_features
from ewt_final.pipeline import load_config


def weekly_panel(end="2015-12-28"):
    dates = pd.date_range("2007-01-01", end, freq="W-MON")
    seasonal = ((np.sin(np.arange(len(dates)) * 2 * np.pi / 52) + 1) * 4).astype(int)
    return pd.DataFrame(
        {
            "municipality_code": "08001",
            "week_start": dates,
            "confirmed_records_screening": seasonal,
            "temperature_c": 25 + np.sin(np.arange(len(dates)) / 20),
            "precipitation_mm": np.maximum(0, np.cos(np.arange(len(dates)) / 10) * 10),
            "relative_humidity_pct": 75,
            "population": 1_000_000,
            "population_density_km2": 5000,
        }
    )


def test_ewt_contract_and_exact_constant():
    assert ewt((7, 7, 7, 7, 7)) == 7.0
    channel = build_ewt_channel(weekly_panel(), EwtConfig(partial_week_starts=()))
    assert {"ch_ewt", "ch_upper", "ch_gsd", "excess_flag_ewt"}.issubset(channel)
    assert channel.loc[channel.week_start.dt.year >= 2012, "ch_ewt"].notna().all()


def test_episode_label_masks_outbreak_and_refractory():
    dates = pd.date_range("2020-01-06", periods=20, freq="W-MON")
    flags = [False] * 5 + [True] * 3 + [False] * 12
    episodes = episodes_from_flags(dates, flags, "08001", min_len=3, merge_gap=0)
    target = label_pre_onset(dates, [True] * len(dates), episodes, horizon=4, refractory=2)
    assert target.iloc[1:5].eq(1).all()
    assert target.iloc[5:10].isna().all()


def test_feature_contract_is_66_and_future_mutation_does_not_change_past():
    source = weekly_panel()
    channel = build_ewt_channel(source, EwtConfig(partial_week_starts=()))
    channel["episode_active"] = False
    before = build_features(channel, horizons=[8], default_horizon=8)
    changed = source.copy()
    cutoff = pd.Timestamp("2014-01-06")
    changed.loc[changed.week_start > cutoff, "confirmed_records_screening"] += 1000
    channel_changed = build_ewt_channel(changed, EwtConfig(partial_week_starts=()))
    channel_changed["episode_active"] = False
    after = build_features(channel_changed, horizons=[8], default_horizon=8)
    assert len(ALL_FEATURES) == 66
    mask = before.week_start <= cutoff
    pd.testing.assert_frame_equal(before.loc[mask, ALL_FEATURES], after.loc[mask, ALL_FEATURES])


def test_config_forbids_stage_or_alias(tmp_path):
    text = open("configs/pipeline.yaml", encoding="utf-8").read().replace("transition_stage: null", "transition_stage: Production")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="transición"):
        load_config(path)

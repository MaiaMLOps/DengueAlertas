"""F1 acceptance: channel product (contrato §1, §2.2, §2.4, §10.2–10.4; especificación §3 and §6.F1).

Literal figures come from especificación §6.F1 and contrato §2.2 (one-week runs per
municipality), computed with docs/forecasting/verificacion/descriptivos_d3.py. Runs, entries
and onsets are recomputed here from target_u1e, independently of the implementation:
- run: maximal block of consecutive eligible weeks with target_u1e = 1;
- entry 0->1: a run that does not start at the first eligible week;
- onset (Umaña et al., U1e): first week of a run of >= 3 weeks preceded by >= 2 weeks with 0.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.acceptance

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = ["municipality_code", "week_start"]
COUNT = "confirmed_records_screening"
SOURCE_COLUMNS = [
    "municipality_code", "municipality", "week_start", "population", "area_km2",
    "population_density_km2", "temperature_c", "precipitation_mm", "relative_humidity_pct",
    "confirmed_records_screening", "no_eligible_records_observed", "week_end_nominal",
    "week_end_effective", "n_days", "partial_week",
]
CHANNEL_COLUMNS = ["epico_central", "epico_upper", "target_u1e", "channel_zone", "channel_eligible"]
CHANNEL_TYPES = {
    "epico_central": pa.float64(),
    "epico_upper": pa.float64(),
    "target_u1e": pa.int8(),
    "channel_zone": pa.string(),
    "channel_eligible": pa.bool_(),
}
ZONES = ("seguridad", "alerta", "epidemia")
# code: (positives, runs, one-week runs, onsets (Umaña, U1e), entries 0->1)
BY_MUNICIPALITY = {
    "08001": (171, 28, 13, 5, 28),  # Barranquilla
    "08758": (204, 46, 21, 7, 46),  # Soledad
    "41001": (152, 26, 10, 6, 26),  # Neiva
    "50001": (165, 32, 14, 7, 31),  # Villavicencio
    "54001": (129, 30, 21, 3, 30),  # Cúcuta
    "76001": (124, 13, 9, 3, 13),   # Cali
}


def _sorted(path):
    return pq.read_table(path).sort_by([(key, "ascending") for key in KEYS])


def _eligible_frame(table):
    frame = table.to_pandas()
    return frame[frame["channel_eligible"]].sort_values(KEYS).reset_index(drop=True)


def _runs(values):
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


def _event_counts(values):
    runs = _runs(values)
    onsets = [start for start, length in runs
              if length >= 3 and start >= 2 and values[start - 1] == 0 and values[start - 2] == 0]
    return {
        "positives": int(sum(values)),
        "runs": len(runs),
        "one_week_runs": sum(1 for _, length in runs if length == 1),
        "onsets": len(onsets),
        "entries": sum(1 for start, _ in runs if start >= 1),
    }


# --------------------------------------------------------------------------- product alone
def test_shape_key_and_no_legacy_columns(channel_path):
    table = _sorted(channel_path)

    assert table.num_rows == 5952
    assert "ewt" not in table.column_names
    assert "excess_label" not in table.column_names
    assert table.column_names == SOURCE_COLUMNS + CHANNEL_COLUMNS
    keys = table.select(KEYS).to_pylist()
    assert len({(row["municipality_code"], row["week_start"]) for row in keys}) == 5952
    codes = table.column("municipality_code").to_pylist()
    assert {code: codes.count(code) for code in BY_MUNICIPALITY} == {code: 992 for code in BY_MUNICIPALITY}


def test_channel_column_types(channel_path):
    schema = pq.read_schema(channel_path)

    for name, expected in CHANNEL_TYPES.items():
        assert schema.field(name).type == expected, f"{name}: {schema.field(name).type} != {expected}"


def test_nulls_follow_eligibility(channel_path):
    frame = _sorted(channel_path).to_pandas()
    eligible = frame["channel_eligible"]

    assert eligible.notna().all()
    for name in ["epico_central", "epico_upper", "target_u1e", "channel_zone"]:
        assert frame.loc[eligible, name].notna().all(), f"{name} null in an eligible week"
        assert frame.loc[~eligible, name].isna().all(), f"{name} present in an ineligible week"


def test_eligible_weeks_and_positives(channel_path):
    eligible = _eligible_frame(_sorted(channel_path))

    assert len(eligible) == 4380
    assert eligible.groupby("municipality_code").size().to_dict() == {code: 730 for code in BY_MUNICIPALITY}
    assert int(eligible["target_u1e"].astype(int).sum()) == 945


def test_target_and_zone_are_consistent_with_thresholds(channel_path):
    eligible = _eligible_frame(_sorted(channel_path))
    count = eligible[COUNT]
    central = eligible["epico_central"]
    upper = eligible["epico_upper"]

    assert (eligible["target_u1e"].astype(int) == (count > upper).astype(int)).all()
    expected_zone = (count > central).astype(int) + (count > upper).astype(int)
    assert (eligible["channel_zone"].map({zone: i for i, zone in enumerate(ZONES)}) == expected_zone).all()


def test_upper_never_below_central(channel_path):
    eligible = _eligible_frame(_sorted(channel_path))

    assert (eligible["epico_upper"] >= eligible["epico_central"]).all()


def test_runs_onsets_and_entries_by_municipality(channel_path):
    eligible = _eligible_frame(_sorted(channel_path))

    observed = {}
    for code, group in eligible.groupby("municipality_code", sort=True):
        counts = _event_counts(group["target_u1e"].astype(int).tolist())
        observed[code] = (counts["positives"], counts["runs"], counts["one_week_runs"],
                          counts["onsets"], counts["entries"])

    assert observed == BY_MUNICIPALITY
    totals = [sum(values[i] for values in observed.values()) for i in range(5)]
    assert totals == [945, 175, 88, 31, 174]


def test_zone_shares(channel_path):
    eligible = _eligible_frame(_sorted(channel_path))
    shares = eligible["channel_zone"].value_counts(normalize=True)

    assert [round(float(shares.get(zone, 0.0)), 3) for zone in ZONES] == [0.379, 0.405, 0.216]
    assert round(float((eligible[COUNT] > eligible["epico_central"]).mean()), 3) == 0.621


# --------------------------------------------------------------------------- against the legacy
def test_source_columns_identical_to_legacy(channel_path, legacy_labels_path):
    channel = _sorted(channel_path)
    legacy = _sorted(legacy_labels_path)

    assert channel.num_rows == legacy.num_rows
    type_differences = {name: (str(channel.schema.field(name).type), str(legacy.schema.field(name).type))
                        for name in SOURCE_COLUMNS
                        if channel.schema.field(name).type != legacy.schema.field(name).type}
    assert type_differences == {}, f"Type differences (channel, legacy): {type_differences}"
    for name in SOURCE_COLUMNS:
        assert channel.column(name).combine_chunks().equals(legacy.column(name).combine_chunks()), name


def test_eligibility_matches_legacy(channel_path, legacy_labels_path):
    channel = _sorted(channel_path)
    legacy = _sorted(legacy_labels_path)

    assert channel.column("channel_eligible").to_pylist() == legacy.column("label_eligible").to_pylist()


def test_central_strictly_above_legacy_ewt(channel_path, legacy_labels_path):
    channel = _sorted(channel_path).to_pandas()
    legacy = _sorted(legacy_labels_path).to_pandas()
    eligible = channel["channel_eligible"].to_numpy()

    above = channel.loc[eligible, "epico_central"].to_numpy() > legacy.loc[eligible, "ewt"].to_numpy()
    assert int(eligible.sum()) == 4380
    assert int(above.sum()) == 4380


# --------------------------------------------------------------------------- code provenance
def test_new_code_does_not_reference_legacy_labels():
    channel_module = REPO_ROOT / "src" / "epidemiology" / "channel_epico.py"
    assert channel_module.is_file(), "src/epidemiology/channel_epico.py is missing"
    files = [channel_module, *sorted((REPO_ROOT / "src" / "forecasting").rglob("*.py"))]

    offenders = [str(path.relative_to(REPO_ROOT)) for path in files
                 if "pilot_weekly_labels" in path.read_text(encoding="utf-8")]
    assert offenders == []

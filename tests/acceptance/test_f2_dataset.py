"""F2 acceptance: supervised examples and the fifteen features (contrato §3.1–3.2, §3.7, §5, §10.5 and
§10.7; especificación §4 «Canal: reparto y reutilización» and §6.F2), with h = 8.

Literal figures:
- rows: 711 per municipality, 4,266 in total, t from 2012-03-19 to 2025-10-27 (especificación §6.F2;
  contrato §5.4);
- three hand-explained cases (Villavicencio 2018-12-03, Barranquilla 2018-11-05 and 2012-12-31) whose
  values Jesús computed outside the repository; compared with pytest.approx(rel=1e-9, abs=1e-12).

Truncation (contrato §10.7). The truncated base for an origin (municipality, t) is the panel's fifteen
source columns of that municipality only, with week_start <= t. It has no channel columns, so central,
U1e (including U1e_{t+h} and central_{t+h}) and clim_s cannot be read from a product computed with the
full base: `build_features` must recompute them from the counts and climate it receives, and no row after
t exists. Features computed with the full base and with the truncated base must be exactly equal. The
sensitivity test closes the remaining gap: altering weeks <= t of the truncated base must change the
features that depend on them, so features do come from the base passed in and not from a file.

`truncation_mismatches` is a module-level helper on purpose: tests/forecasting imports it by path to
check, with deliberately leaky builders, that this frozen comparison fails when a feature leaks.

The oracle recomputes the fifteen features in pandas, independently of the implementation, from the
channel product (its epico_* columns and climate), with its own ordinal alignment (53 -> 52 year by
year) and the literal windows of contrato §5.2.
"""

from datetime import date, timedelta
import functools
import hashlib
import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.acceptance

H = 8                                     # especificación §6.F2
COUNT = "confirmed_records_screening"
CODES = ("08001", "08758", "41001", "50001", "54001", "76001")
ZONES = ("seguridad", "alerta", "epidemia")
CONTRACT_FEATURES = (                     # contrato §5.2, in table order
    "zona_t", "posicion_4s", "crecimiento_4s", "cambio_umbral", "sem_sin", "sem_cos",
    "anom_temp_m1", "anom_temp_m2", "anom_temp_m3",
    "anom_prec_m1", "anom_prec_m2", "anom_prec_m3",
    "anom_hum_m1", "anom_hum_m2", "anom_hum_m3",
)
KEY_COLUMNS = ("municipality_code", "t")
EXAMPLE_COLUMNS = ["municipality_code", "t", "target_week", "y", *CONTRACT_FEATURES]

FIRST_T = date(2012, 3, 19)               # contrato §5.4
LAST_T = date(2025, 10, 27)               # contrato §3.7 and §5.4
ROWS_PER_MUNICIPALITY = 711
ALL_ORIGINS = tuple(FIRST_T + timedelta(weeks=i) for i in range(ROWS_PER_MUNICIPALITY))
SPECIAL_ORIGINS = (
    date(2012, 11, 5), date(2018, 11, 5), date(2024, 11, 4),     # t + 8 has ordinal 53
    date(2012, 12, 31), date(2018, 12, 31), date(2024, 12, 30),  # t has ordinal 53 (clim 53 -> 52)
    date(2018, 12, 3),                                           # literal case A
)
SENSITIVITY_ORIGINS = (date(2014, 6, 2), date(2018, 12, 3), date(2023, 4, 3))

# Literal cases (Jesús, computed outside the repository).
LITERAL_TOLERANCE = {"rel": 1e-9, "abs": 1e-12}
LITERAL_CASES = {
    "A": {
        "code": "50001", "t": date(2018, 12, 3), "count_t": 28,
        "central_t": 2.75848, "upper_t": 9.889469222544161,
        "central_target": 8.73763, "upper_target": 38.46939411375516,
        "features": {
            "zona_t": "epidemia",
            "posicion_4s": 0.20920084516145931,
            "crecimiento_4s": 0.534082485930258,
            "cambio_umbral": 1.2877293431439067,
            "sem_sin": 0.35343541937821843,
            "sem_cos": 0.9354589271202359,
            "anom_temp_m1": 0.004798298382446831,
            "anom_prec_m1": -153.28622195414795,
            "anom_hum_m1": 1.5250616556161347,
            "anom_temp_m2": -0.31994891129978953,
            "anom_prec_m2": -28.759252804069973,
            "anom_hum_m2": 2.424319120178623,
            "anom_temp_m3": -0.3968445311525377,
            "anom_prec_m3": 137.5830691279357,
            "anom_hum_m3": 3.1897683238875736,
        },
    },
    "B": {
        "code": "08001", "t": date(2018, 11, 5), "count_t": 39,
        "central_t": 34.60812, "upper_t": 188.22369333622296,
        "central_target": 30.29914, "upper_target": 170.11691817360386,
        "features": {
            "zona_t": "alerta",
            "posicion_4s": -1.5595846903137867,
            "crecimiento_4s": 0.5108256237659905,
            "cambio_umbral": -0.10058282295327992,
            "sem_sin": -0.021672763616436087,
            "sem_cos": 0.9997651180738534,
            "anom_temp_m1": 0.2569534944028602,
            "anom_prec_m1": -11.147922349595941,
            "anom_hum_m1": -1.153413397861005,
            "anom_temp_m2": -0.00262168832372911,
            "anom_prec_m2": -33.54950472264624,
            "anom_hum_m2": -0.776560706974788,
            "anom_temp_m3": 0.18275098606030937,
            "anom_prec_m3": -8.001283766223303,
            "anom_hum_m3": -1.2417517729457401,
        },
    },
    "C": {
        "code": "08001", "t": date(2012, 12, 31), "count_t": 50,
        "central_t": 24.94542, "upper_t": 91.08973517464791,
        "central_target": 10.89757, "upper_target": 70.48021156104373,
        "features": {
            "zona_t": "alerta",
            "posicion_4s": -0.9429368597278273,
            "crecimiento_4s": 0.6359887667199966,
            "cambio_umbral": -0.253342833995454,
            "sem_sin": 0.7465727613654504,
            "sem_cos": 0.6653037742168356,
            "anom_temp_m1": 0.9725639077270998,
            "anom_prec_m1": -6.560738810172288,
            "anom_hum_m1": -2.1193397394531637,
            "anom_temp_m2": 0.5219559624081764,
            "anom_prec_m2": -40.239311843471775,
            "anom_hum_m2": -1.9293928831437448,
            "anom_temp_m3": 0.6270621500858837,
            "anom_prec_m3": -77.20643426641378,
            "anom_hum_m3": -2.287620884293826,
        },
    },
}

# Exact equality when a function is compared with itself (truncation, channel_for_week against the
# product); tolerance when two implementations are compared (oracle, literals).
# Oracle constants, literal from contrato §5.2 (not read from the implementation or configuration).
ORACLE_TOLERANCE = {"rel": 1e-9, "abs": 1e-12}
SEM_PERIOD = 52.18
ORACLE_BLOCKS = {"m1": range(0, 4), "m2": range(4, 8), "m3": range(8, 12)}  # weeks before t
CLIMATE = {"temp": ("temperature_c", "mean"), "prec": ("precipitation_mm", "sum"),
           "hum": ("relative_humidity_pct", "mean")}


# --------------------------------------------------------------------------- helpers
def _config():
    from src.forecasting.config import load_config

    return load_config()


def _build_features(config):
    from src.forecasting.dataset import build_features

    return functools.partial(build_features, h=H, canal=config.canal, features=config.features)


def _channel_kwargs(config):
    return {"n_years": config.canal.n, "shift": config.canal.desplazamiento, "alpha": config.canal.cuantil_t}


def _as_date(value):
    return pd.Timestamp(value).date()


def _same(a, b):
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def sample_origins():
    """At least 50 origins spread over the whole range (first and last included) plus special weeks."""
    spread = {ALL_ORIGINS[i] for i in np.linspace(0, ROWS_PER_MUNICIPALITY - 1, 50).round().astype(int)}
    return sorted(spread | set(SPECIAL_ORIGINS))


def truncation_mismatches(build, source: pa.Table, origins):
    """Differences between features built with the full `source` and with `source` cut at each t.

    `build(base, origins) -> DataFrame` with columns municipality_code, t and feature columns. The cut
    base keeps only the origin's municipality and its weeks <= t. The compared columns are those the
    full build returns besides the keys; they must not be empty and must be contract features.
    """
    origins = [(code, _as_date(t)) for code, t in origins]
    full = build(source, origins)
    columns = [name for name in full.columns if name not in KEY_COLUMNS]
    assert columns, "The builder returned no feature columns to compare."
    outside = sorted(set(columns) - set(CONTRACT_FEATURES))
    assert not outside, f"Compared columns outside the contract features: {outside}"
    full_rows = {(code, _as_date(t)): row for code, t, row in
                 zip(full["municipality_code"], full["t"], full[columns].to_dict("records"))}
    absent = [origin for origin in origins if origin not in full_rows]
    assert not absent, f"Full-base build lacks origins: {absent[:5]}"

    codes, weeks = source.column("municipality_code"), source.column("week_start")
    mismatches = []
    for code, t in origins:
        cut = source.filter(pc.and_(pc.equal(codes, code), pc.less_equal(weeks, pa.scalar(t, pa.date32()))))
        assert cut.num_rows > 0 and pc.max(cut.column("week_start")).as_py() == t, f"Cut base for {code} {t} does not end at t."
        truncated = build(cut, [(code, t)])
        lacking = [name for name in columns if name not in truncated.columns]
        if lacking:
            mismatches.append(f"{code} {t}: truncated build lacks {lacking}")
            continue
        rows = {(c, _as_date(s)): row for c, s, row in
                zip(truncated["municipality_code"], truncated["t"], truncated[columns].to_dict("records"))}
        if (code, t) not in rows:
            mismatches.append(f"{code} {t}: truncated build lacks the origin row")
            continue
        for name in columns:
            full_value, cut_value = full_rows[(code, t)][name], rows[(code, t)][name]
            if not _same(full_value, cut_value):
                mismatches.append(f"{code} {t} {name}: full={full_value!r} truncated={cut_value!r}")
    return mismatches


def _complete_counts(table: pa.Table, code: str, last: date | None = None) -> dict[date, int]:
    """Counts of the complete weeks of one municipality, optionally up to `last` inclusive."""
    return {row["week_start"]: row[COUNT]
            for row in table.select(["municipality_code", "week_start", "partial_week", COUNT]).to_pylist()
            if row["municipality_code"] == code and not row["partial_week"]
            and (last is None or row["week_start"] <= last)}


def _single_municipality_cut(table: pa.Table, code: str, t: date) -> pa.Table:
    return table.filter(pc.and_(pc.equal(table.column("municipality_code"), code),
                                pc.less_equal(table.column("week_start"), pa.scalar(t, pa.date32()))))


def _modified(table: pa.Table, week: date, changes: dict) -> pa.Table:
    """Copy of a single-municipality table with `changes` applied to the row of `week`."""
    rows = table.to_pylist()
    hits = 0
    for row in rows:
        if row["week_start"] == week:
            for name, change in changes.items():
                row[name] = change(row[name])
            row["no_eligible_records_observed"] = row[COUNT] == 0
            hits += 1
    assert hits == 1, f"Expected one row for {week}; found {hits}."
    return pa.Table.from_pylist(rows, schema=table.schema)


def _ordinal(start: date) -> int:
    first = date(start.year, 1, 1)
    first_monday = first + timedelta(days=(-first.weekday()) % 7)
    return (start - first_monday).days // 7 + 1


class _Oracle:
    """Independent recomputation of the fifteen features from the channel product."""

    def __init__(self, product: pa.Table):
        self.by_week, self.by_ordinal = {}, {}
        for row in product.to_pylist():
            start = row["week_start"]
            self.by_week[(row["municipality_code"], start)] = row
            self.by_ordinal[(row["municipality_code"], start.year, _ordinal(start))] = row

    def references(self, code: str, start: date) -> list[dict]:
        ordinal, rows = _ordinal(start), []
        for year in range(start.year - 5, start.year):
            row = self.by_ordinal.get((code, year, ordinal))
            if row is None and ordinal == 53:
                row = self.by_ordinal.get((code, year, 52))
            assert row is not None and not row["partial_week"], f"No reference for {code} {start} in {year}."
            rows.append(row)
        return rows

    def anomaly(self, code: str, start: date, column: str) -> float:
        references = self.references(code, start)
        return self.by_week[(code, start)][column] - sum(row[column] for row in references) / len(references)

    def features(self, code: str, t: date) -> dict:
        row = lambda s: self.by_week[(code, s)]  # noqa: E731
        weeks = [t - timedelta(weeks=k) for k in range(12)]
        target = t + timedelta(weeks=H)
        log1 = lambda x: math.log(x + 1)  # noqa: E731
        angle = 2 * math.pi * (_ordinal(target) - 1) / SEM_PERIOD
        values = {
            "zona_t": row(t)["channel_zone"],
            "posicion_4s": sum(log1(row(s)[COUNT]) - log1(row(s)["epico_upper"]) for s in weeks[0:4]) / 4,
            "crecimiento_4s": log1(row(t)[COUNT]) - log1(row(weeks[4])[COUNT]),
            "cambio_umbral": log1(row(target)["epico_upper"]) - log1(row(t)["epico_upper"]),
            "sem_sin": math.sin(angle),
            "sem_cos": math.cos(angle),
        }
        for short, (column, how) in CLIMATE.items():
            for block, offsets in ORACLE_BLOCKS.items():
                anomalies = [self.anomaly(code, weeks[k], column) for k in offsets]
                values[f"anom_{short}_{block}"] = sum(anomalies) / len(anomalies) if how == "mean" else sum(anomalies)
        return values


def _assert_feature_values(observed: dict, expected: dict, tolerance: dict, label: str):
    for name in CONTRACT_FEATURES:
        if name == "zona_t":
            assert observed[name] == expected[name], f"{label} zona_t: {observed[name]!r} != {expected[name]!r}"
        else:
            assert observed[name] == pytest.approx(expected[name], **tolerance), \
                f"{label} {name}: {observed[name]!r} != {expected[name]!r}"


@functools.lru_cache(maxsize=None)
def _examples(channel_file: str):
    from src.forecasting.dataset import build_examples, load_channel_product

    config = _config()
    product = load_channel_product(channel_file, expected_md5=config.datos.version_dvc_canal)
    return build_examples(product, h=H, canal=config.canal, features=config.features)


# --------------------------------------------------------------------------- API and rows
def test_feature_names_match_contract():
    from src.forecasting.dataset import FEATURES

    assert tuple(FEATURES) == CONTRACT_FEATURES


def test_rows_per_municipality_and_origin_range(channel_path):
    examples = _examples(str(channel_path))

    assert list(examples.columns) == EXAMPLE_COLUMNS
    assert len(examples) == 4266
    assert examples.groupby("municipality_code").size().to_dict() == {code: 711 for code in CODES}
    for code, group in examples.groupby("municipality_code"):
        origins = sorted(_as_date(t) for t in group["t"])
        assert origins[0] == FIRST_T and origins[-1] == LAST_T, code
        assert origins == list(ALL_ORIGINS), code
    assert not examples.duplicated(list(KEY_COLUMNS)).any()


def test_target_week_is_t_plus_h_eligible_and_labelled(channel_path):
    examples = _examples(str(channel_path))
    product = {(row["municipality_code"], row["week_start"]): row for row in pq.read_table(
        channel_path, columns=["municipality_code", "week_start", "channel_eligible", "target_u1e"]).to_pylist()}

    for code, t, target, y in zip(examples["municipality_code"], examples["t"], examples["target_week"], examples["y"]):
        t, target = _as_date(t), _as_date(target)
        assert target == t + timedelta(weeks=H), (code, t)
        assert product[(code, target)]["channel_eligible"], (code, t)
        assert int(y) == product[(code, target)]["target_u1e"], (code, t)


def test_fifteen_features_without_missing_values(channel_path):
    examples = _examples(str(channel_path))

    assert [name for name in examples.columns if name in CONTRACT_FEATURES] == list(CONTRACT_FEATURES)
    assert not examples[list(CONTRACT_FEATURES)].isna().any().any()
    assert set(examples["zona_t"]) <= set(ZONES)
    numeric = examples[list(CONTRACT_FEATURES[1:])].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


# --------------------------------------------------------------------------- channel outside the panel
def test_channel_for_week_reproduces_product(panel_path, channel_path):
    from src.epidemiology.channel_epico import channel_for_week

    kwargs = _channel_kwargs(_config())
    panel = pq.read_table(panel_path)
    product = pq.read_table(channel_path, columns=[
        "municipality_code", "week_start", "channel_eligible", "epico_central", "epico_upper"]).to_pylist()
    counts = {code: _complete_counts(panel, code) for code in CODES}

    eligible = initialization = 0
    for row in product:
        result = channel_for_week(counts[row["municipality_code"]], row["week_start"], **kwargs)
        if row["channel_eligible"]:
            eligible += 1
            assert result == (row["epico_central"], row["epico_upper"]), (row["municipality_code"], row["week_start"])
        elif row["week_start"].year < 2012:
            initialization += 1
            assert result is None, (row["municipality_code"], row["week_start"])
    assert (eligible, initialization) == (4380, 1566)


def test_target_week_channel_unchanged_by_truncation(panel_path, channel_path):
    from src.epidemiology.channel_epico import channel_for_week

    kwargs = _channel_kwargs(_config())
    panel = pq.read_table(panel_path)
    product = {(row["municipality_code"], row["week_start"]): row for row in pq.read_table(
        channel_path, columns=["municipality_code", "week_start", "epico_central", "epico_upper"]).to_pylist()}

    checked = 0
    for code in CODES:
        complete = _complete_counts(panel, code)
        for t in ALL_ORIGINS:
            target = t + timedelta(weeks=H)
            truncated = {week: count for week, count in complete.items() if week <= t}
            expected = product[(code, target)]
            assert channel_for_week(truncated, target, **kwargs) == (expected["epico_central"], expected["epico_upper"]), (code, t)
            checked += 1
    assert checked == 4266


# --------------------------------------------------------------------------- truncation and sensitivity
def test_truncation_sample_has_spread_and_special_origins():
    origins = sample_origins()

    assert len(origins) >= 50
    assert origins[0] == FIRST_T and origins[-1] == LAST_T
    assert set(SPECIAL_ORIGINS) <= set(origins)
    assert {case["t"] for case in LITERAL_CASES.values()} <= set(origins)
    assert set(origins) <= set(ALL_ORIGINS)


def test_features_identical_with_full_and_truncated_base(panel_path):
    build = _build_features(_config())
    panel = pq.read_table(panel_path)
    origins = [(code, t) for code in CODES for t in sample_origins()]

    assert truncation_mismatches(build, panel, origins) == []


@pytest.mark.parametrize("t", SENSITIVITY_ORIGINS, ids=str)
@pytest.mark.parametrize("code", CODES)
def test_features_respond_to_the_base_passed_in(panel_path, code, t):
    build = _build_features(_config())
    cut = _single_municipality_cut(pq.read_table(panel_path), code, t)
    target = t + timedelta(weeks=H)
    ordinal = _ordinal(target)
    reference_year = target.year - 1
    references = [week for week in cut.column("week_start").to_pylist()
                  if week.year == reference_year and _ordinal(week) == ordinal]
    if not references and ordinal == 53:
        references = [week for week in cut.column("week_start").to_pylist()
                      if week.year == reference_year and _ordinal(week) == 52]
    assert len(references) == 1

    def features(table):
        frame = build(table, [(code, t)])
        return frame.iloc[0].to_dict()

    base = features(cut)
    count_at_t = features(_modified(cut, t, {COUNT: lambda c: c + 100}))
    reference = features(_modified(cut, references[0], {COUNT: lambda c: c + 100}))
    climate = features(_modified(cut, t, {"temperature_c": lambda x: x + 5.0,
                                          "precipitation_mm": lambda x: x + 50.0,
                                          "relative_humidity_pct": lambda x: x + 5.0}))

    assert count_at_t["crecimiento_4s"] != base["crecimiento_4s"]
    assert reference["cambio_umbral"] != base["cambio_umbral"]
    for name in ("anom_temp_m1", "anom_prec_m1", "anom_hum_m1"):
        assert climate[name] != base[name], name


# --------------------------------------------------------------------------- literal cases
@pytest.mark.parametrize("case", sorted(LITERAL_CASES))
def test_literal_case_counts_in_panel(panel_path, case):
    spec = LITERAL_CASES[case]

    assert _complete_counts(pq.read_table(panel_path), spec["code"])[spec["t"]] == spec["count_t"]


@pytest.mark.parametrize("case", sorted(LITERAL_CASES))
def test_literal_case_channel(panel_path, case):
    from src.epidemiology.channel_epico import channel_for_week

    spec = LITERAL_CASES[case]
    kwargs = _channel_kwargs(_config())
    counts = _complete_counts(pq.read_table(panel_path), spec["code"])

    central_t, upper_t = channel_for_week(counts, spec["t"], **kwargs)
    central_target, upper_target = channel_for_week(counts, spec["t"] + timedelta(weeks=H), **kwargs)
    assert central_t == pytest.approx(spec["central_t"], **LITERAL_TOLERANCE)
    assert upper_t == pytest.approx(spec["upper_t"], **LITERAL_TOLERANCE)
    assert central_target == pytest.approx(spec["central_target"], **LITERAL_TOLERANCE)
    assert upper_target == pytest.approx(spec["upper_target"], **LITERAL_TOLERANCE)


@pytest.mark.parametrize("case", sorted(LITERAL_CASES))
def test_literal_case_features(panel_path, case):
    spec = LITERAL_CASES[case]
    frame = _build_features(_config())(pq.read_table(panel_path), [(spec["code"], spec["t"])])

    assert len(frame) == 1
    _assert_feature_values(frame.iloc[0].to_dict(), spec["features"], LITERAL_TOLERANCE, f"case {case}")


# --------------------------------------------------------------------------- independent oracle
@pytest.mark.parametrize("case", sorted(LITERAL_CASES))
def test_oracle_reproduces_literal_cases(channel_path, case):
    """Checks the oracle itself against Jesús's independent values; it does not use the implementation."""
    spec = LITERAL_CASES[case]

    _assert_feature_values(_Oracle(pq.read_table(channel_path)).features(spec["code"], spec["t"]),
                           spec["features"], LITERAL_TOLERANCE, f"oracle case {case}")


def test_features_match_oracle_on_sample(channel_path):
    examples = _examples(str(channel_path))
    oracle = _Oracle(pq.read_table(channel_path))
    rows = {(code, _as_date(t)): row for code, t, row in
            zip(examples["municipality_code"], examples["t"], examples.to_dict("records"))}

    for code in CODES:
        for t in sample_origins():
            _assert_feature_values(rows[(code, t)], oracle.features(code, t), ORACLE_TOLERANCE, f"{code} {t}")

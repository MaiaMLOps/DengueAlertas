"""Validate this panel's physical contract without repairing its data."""

from datetime import date, timedelta
import math

import pyarrow as pa
import pyarrow.compute as pc

from .config import ADDED_COLUMNS, COUNT, FIRST_LABEL_YEAR, EpidemiologyValidationError

SOURCE_SCHEMA = pa.schema([
    ("municipality_code", pa.string()), ("municipality", pa.string()),
    ("week_start", pa.date32()), ("population", pa.int64()),
    ("area_km2", pa.float64()), ("population_density_km2", pa.float64()),
    ("temperature_c", pa.float64()), ("precipitation_mm", pa.float64()),
    ("relative_humidity_pct", pa.float64()), (COUNT, pa.int64()),
    ("no_eligible_records_observed", pa.bool_()), ("week_end_nominal", pa.date32()),
    ("week_end_effective", pa.date32()), ("n_days", pa.uint8()), ("partial_week", pa.bool_()),
])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EpidemiologyValidationError(message)


def validate_source(table: pa.Table) -> None:
    require(table.num_rows > 0, "Source panel is empty.")
    require(table.column_names == SOURCE_SCHEMA.names, "Expected the fifteen original columns in order.")
    require(all(a.type == b.type for a, b in zip(table.schema, SOURCE_SCHEMA)),
            "Original Arrow types disagree with the panel contract.")
    require(all(column.null_count == 0 for column in table.columns), "Source panel contains nulls.")
    for field in SOURCE_SCHEMA:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            require(pc.all(pc.is_finite(table[field.name])).as_py(), "Nonfinite source numeric values.")
    rows = table.to_pylist()
    keys = [(r["municipality_code"], r["week_start"]) for r in rows]
    require(len(set(keys)) == len(keys), "Duplicate municipality-week keys.")
    calendars: dict[str, list[date]] = {}
    names: dict[str, set[str]] = {}
    end = date(2025, 12, 31)
    for row in rows:
        code, start = row["municipality_code"], row["week_start"]
        require(len(code) == 5 and code.isascii() and code.isdigit(), "DIVIPOLA must be five text digits.")
        require(bool(row["municipality"].strip()), "Empty municipal name.")
        require(start.weekday() == 0 and date(2007, 1, 1) <= start <= end,
                "Expected Mondays inside 2007-2025.")
        nominal = start + timedelta(days=6)
        effective = min(nominal, end)
        days = (effective - start).days + 1
        require(row["week_end_nominal"] == nominal and row["week_end_effective"] == effective
                and row["n_days"] == days and row["partial_week"] == (days < 7),
                "Inconsistent calendar coverage or partial-week indicator.")
        require(row[COUNT] >= 0, "Negative source count.")
        require(row["no_eligible_records_observed"] == (row[COUNT] == 0), "Inconsistent zero-record indicator.")
        calendars.setdefault(code, []).append(start)
        names.setdefault(code, set()).add(row["municipality"])
    last = max(start for _, start in keys)
    expected = []
    current = date(2007, 1, 1)
    while current <= last:
        expected.append(current)
        current += timedelta(days=7)
    require(all(sorted(values) == expected for values in calendars.values()),
            "Missing weeks, unequal municipal calendars or incomplete initialization period.")
    require(all(len(values) == 1 for values in names.values()), "Conflicting municipal names.")


def validate_product(source: pa.Table, product: pa.Table) -> dict:
    require(product.column_names == source.column_names + ADDED_COLUMNS, "Unexpected product columns.")
    require(product.select(source.column_names).equals(source, check_metadata=False),
            "Original row order, values or Arrow fields changed.")
    require(product.schema.field("ewt").type == pa.float64()
            and product.schema.field("excess_label").type == pa.int8()
            and product.schema.field("label_eligible").type == pa.bool_(), "Unexpected label types.")
    require(product["label_eligible"].null_count == 0, "Eligibility contains nulls.")
    eligible = initialization = partial = 0
    by_municipality = {}
    for row in product.to_pylist():
        code = row["municipality_code"]
        summary = by_municipality.setdefault(code, {"rows": 0, "eligible": 0, "ineligible": 0})
        summary["rows"] += 1
        initial = row["week_start"].year < FIRST_LABEL_YEAR
        initialization += initial
        partial += row["partial_week"]
        if row["label_eligible"]:
            require(not initial and not row["partial_week"], "Initialization or partial week was labelled.")
            require(row["ewt"] is not None and math.isfinite(row["ewt"]) and row["ewt"] >= 0,
                    "Eligible EWT must be finite and nonnegative.")
            require(row["excess_label"] == int(row[COUNT] > row["ewt"]), "Strict comparison disagrees.")
            eligible += 1
            summary["eligible"] += 1
        else:
            require(row["ewt"] is None and row["excess_label"] is None, "Ineligible fields must both be null.")
            summary["ineligible"] += 1
    require(product["ewt"].null_count == product.num_rows - eligible
            and product["excess_label"].null_count == product.num_rows - eligible,
            "Null masks disagree with eligibility.")
    return {"rows": product.num_rows, "columns": product.num_columns, "eligible": eligible,
            "ineligible": product.num_rows - eligible, "initialization_rows": initialization,
            "partial_rows": partial, "original_columns_preserved": True,
            "by_municipality": dict(sorted(by_municipality.items()))}

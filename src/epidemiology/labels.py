"""Append weekly states without changing original Arrow arrays or row order."""

import json

import pyarrow as pa

from .config import COUNT, method_contract
from .ewt import calculate_ewt
from .validation import validate_product


def label_weeks(source: pa.Table) -> tuple[pa.Table, dict]:
    thresholds, audit = calculate_ewt(source)
    counts = source[COUNT].to_pylist()
    eligible = [value is not None for value in thresholds]
    labels = [int(count > threshold) if threshold is not None else None
              for count, threshold in zip(counts, thresholds)]
    result = source.append_column("ewt", pa.array(thresholds, type=pa.float64()))
    result = result.append_column("excess_label", pa.array(labels, type=pa.int8()))
    result = result.append_column(pa.field("label_eligible", pa.bool_(), nullable=False), pa.array(eligible))
    # Source-wide metadata may explicitly describe an unlabelled product. Keep its
    # screening rule and extend its dictionary, but describe this new product anew.
    original = source.schema.metadata or {}
    dictionary = json.loads(original.get(b"denguealertas.data_dictionary", b"{}"))
    dictionary.update({
        "ewt": "Shifted geometric mean of five prior-year aligned weekly counts; null when ineligible.",
        "excess_label": "1 iff count > unrounded ewt; 0 otherwise if eligible; null when ineligible.",
        "label_eligible": "True only for complete target weeks from 2012 with five valid historical references.",
    })
    metadata = {
        b"denguealertas.data_dictionary": json.dumps(dictionary, ensure_ascii=False).encode("utf-8"),
        b"denguealertas.epidemiology": json.dumps(method_contract(), sort_keys=True).encode("utf-8"),
        b"denguealertas.warnings": json.dumps([
            "Retrospective weekly excess of a historical central reference, not an officially confirmed outbreak.",
            "Final event-210 confirmed records; historical availability is not established; zeros are not verified negative notifications.",
            "Original anomalies and absence of person/episode deduplication are retained.",
            "Pilots were selected retrospectively; W-SUN equivalence to the official INS calendar is not established.",
            "No forecast, horizon, supervised features or persistent episode is produced.",
        ]).encode("utf-8"),
    }
    if b"denguealertas.rule" in original:
        metadata[b"denguealertas.rule"] = original[b"denguealertas.rule"]
    result = result.replace_schema_metadata(metadata)
    return result, {"validation": validate_product(source, result), "references": audit}

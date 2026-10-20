"""Stage and verify complete artifacts before publishing without replacement.

The passed report is the completion marker for the pair. Two paths cannot be
published as a single filesystem transaction. Ordinary publication errors roll
back our dataset; abrupt process termination may leave a complete orphan dataset.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .sources import sha256
from .validation import require


@contextmanager
def staged_path(destination: Path) -> Iterator[Path]:
    """Create an owned temporary file on the destination filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(filename)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def publish_exclusive(temporary: Path, destination: Path) -> None:
    """Atomically create a new name for a closed, fully written file.

    Hard links never replace an existing destination, on Windows or POSIX.
    The temporary is on the same filesystem. Unsupported filesystems fail
    explicitly; there is no fallback to writing the final file directly.
    """
    os.link(temporary, destination)


def _serialize_report(temporary: Path, report: dict) -> None:
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda value: value.item())
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_report(destination: Path, report: dict) -> None:
    """Publish an aggregate-only report without partial final JSON or overwrite."""
    with staged_path(destination) as temporary:
        _serialize_report(temporary, report)
        publish_exclusive(temporary, destination)


def publish_panel(table: pa.Table, output: Path, report_path: Path, report: dict) -> None:
    """Verify the staged Parquet and stage its report before publishing either."""
    with staged_path(output) as data_temporary, staged_path(report_path) as report_temporary:
        with data_temporary.open("wb") as stream:
            pq.write_table(table, stream, compression="zstd")
            stream.flush()
            os.fsync(stream.fileno())
        reread = pq.read_table(data_temporary)
        require(reread.equals(table, check_metadata=False), "Parquet readback differs from the validated panel.")
        report["output"] = {
            "file": report["configuration"]["output"], "rows": reread.num_rows,
            "columns": reread.num_columns, "bytes": data_temporary.stat().st_size,
            "sha256": sha256(data_temporary), "readback_equal": True,
            "schema": {f.name: str(f.type) for f in reread.schema},
        }
        report["status"] = "passed"
        _serialize_report(report_temporary, report)
        published_output = False
        try:
            publish_exclusive(data_temporary, output)
            published_output = True
            # Publish last: a passed report with matching checksum marks completion.
            publish_exclusive(report_temporary, report_path)
        except BaseException:
            if published_output and output.exists() and output.samefile(data_temporary):
                output.unlink()
            raise

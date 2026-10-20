import pandas as pd

from src.data.calendar import build_calendar


def test_full_calendar_and_three_day_final_week():
    frame = build_calendar(["08001", "05001"])
    assert len(frame) == 1984
    assert frame.groupby("municipality_code").size().eq(992).all()
    assert not frame.duplicated(["municipality_code", "week_start"]).any()
    partial = frame.loc[frame["partial_week"]]
    assert len(partial) == 2
    assert partial["week_start"].eq(pd.Timestamp("2025-12-29")).all()
    assert partial["week_end_nominal"].eq(pd.Timestamp("2026-01-04")).all()
    assert partial["week_end_effective"].eq(pd.Timestamp("2025-12-31")).all()
    assert partial["n_days"].eq(3).all()
    assert frame.loc[~frame["partial_week"], "n_days"].eq(7).all()


def test_inclusive_calendar_intersection():
    frame = build_calendar(["05001"], "2025-12-31", "2026-01-05")
    assert frame["n_days"].tolist() == [5, 1]
    assert frame["partial_week"].all()

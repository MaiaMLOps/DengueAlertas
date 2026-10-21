"""Descriptive tables from existing weekly labels; no target recalculation."""

import math

import pandas as pd
import pyarrow as pa

from .config import COUNT
from .validation import SOURCE_SCHEMA, require, validate_product, validate_source

RUN_COLUMNS = ['municipality_code', 'municipality', 'start_week', 'end_week',
               'duration_weeks', 'left_boundary', 'right_boundary']


def prepare_labels(table: pa.Table) -> pd.DataFrame:
    source = table.select(SOURCE_SCHEMA.names)
    validate_source(source)
    validate_product(source, table)
    frame = table.to_pandas()
    frame['week_start'] = pd.to_datetime(frame['week_start'])
    frame['excess_label'] = frame['excess_label'].astype('Int8')
    return frame.sort_values(['municipality_code', 'week_start']).reset_index(drop=True)


def validate_label_frame(frame: pd.DataFrame) -> None:
    """Minimal contract also usable with short synthetic sequences and gaps."""
    required = {'municipality_code', 'municipality', 'week_start', 'excess_label', 'label_eligible'}
    require(required.issubset(frame.columns) and len(frame) > 0, 'Missing descriptive label fields or empty input.')
    require(pd.api.types.is_bool_dtype(frame.label_eligible) and frame.label_eligible.notna().all(),
            'Eligibility must be non-null boolean.')
    require(frame.municipality_code.map(lambda code: isinstance(code, str) and len(code) == 5
                                      and code.isascii() and code.isdigit()).all(), 'Invalid DIVIPOLA text.')
    require(frame.municipality.notna().all() and frame.groupby('municipality_code').municipality.nunique().eq(1).all(),
            'Missing or conflicting municipal names.')
    require(pd.api.types.is_datetime64_any_dtype(frame.week_start), 'Expected typed descriptive dates.')
    require(frame.week_start.notna().all() and frame.week_start.dt.tz is None
            and frame.week_start.eq(frame.week_start.dt.normalize()).all()
            and frame.week_start.dt.weekday.eq(0).all(), 'Expected timezone-free Mondays.')
    require(not frame.duplicated(['municipality_code', 'week_start']).any(), 'Duplicate descriptive keys.')
    require(frame.loc[frame.label_eligible, 'excess_label'].isin([0, 1]).all()
            and frame.loc[~frame.label_eligible, 'excess_label'].isna().all(),
            'Eligible labels must be binary; ineligible labels must be null.')


def _groups(frame: pd.DataFrame, columns: list[str]):
    if not columns:
        return [((), frame)]
    return frame.groupby(columns, sort=True, dropna=False)


def label_counts(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in _groups(frame, groups):
        values = key if isinstance(key, tuple) else (key,)
        eligible = group.loc[group.label_eligible, 'excess_label']
        positive, negative = int(eligible.eq(1).sum()), int(eligible.eq(0).sum())
        require(positive + negative == len(eligible), 'Label counts do not reconcile.')
        rows.append({**dict(zip(groups, values)), 'total_weeks': len(group),
                     'eligible_weeks': len(eligible), 'ineligible_weeks': len(group) - len(eligible),
                     'positive_weeks': positive, 'negative_weeks': negative,
                     'positive_pct': 100 * positive / len(eligible) if len(eligible) else math.nan,
                     'negative_pct': 100 * negative / len(eligible) if len(eligible) else math.nan})
    return pd.DataFrame(rows)


def descriptive_statistics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in _groups(frame, groups):
        values = key if isinstance(key, tuple) else (key,)
        eligible = group.loc[group.label_eligible]
        for variable in [COUNT, 'ewt']:
            summary = eligible[variable].describe(percentiles=[.25, .5, .75])
            rows.append({**dict(zip(groups, values)), 'variable': variable,
                         **{name: summary[name] for name in ['count', 'mean', 'std', 'min']},
                         'p25': summary['25%'], 'median': summary['50%'], 'p75': summary['75%'],
                         'max': summary['max']})
    return pd.DataFrame(rows)


def positive_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Maximal observed positive sequences; all rows are visited, including nulls."""
    validate_label_frame(frame)
    result = []
    for (code, name), group in frame.groupby(['municipality_code', 'municipality'], sort=True):
        current = None
        previous = None
        for row in group.sort_values('week_start').itertuples(index=False):
            positive = row.label_eligible and row.excess_label == 1
            gap = previous is not None and row.week_start - previous.week_start != pd.Timedelta(days=7)
            if current is not None and (not positive or gap):
                reason = 'calendar_gap' if gap else ('ineligible_week' if not row.label_eligible else 'negative_week')
                result.append({**current, 'right_boundary': reason})
                current = None
            if positive:
                if current is None:
                    left = ('series_start' if previous is None else 'calendar_gap' if gap
                            else 'ineligible_week' if not previous.label_eligible else 'negative_week')
                    current = {'municipality_code': code, 'municipality': name,
                               'start_week': row.week_start, 'end_week': row.week_start,
                               'duration_weeks': 1, 'left_boundary': left}
                else:
                    current['end_week'] = row.week_start
                    current['duration_weeks'] += 1
            previous = row
        if current is not None:
            result.append({**current, 'right_boundary': 'series_end'})
    return pd.DataFrame(result, columns=RUN_COLUMNS)


def run_summary(frame: pd.DataFrame, runs: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in _groups(frame, groups):
        values = key if isinstance(key, tuple) else (key,)
        selected = runs
        for column, value in zip(groups, values):
            selected = selected.loc[selected[column].eq(value)]
        lengths = selected.duration_weeks.astype('int64')
        rows.append({**dict(zip(groups, values)), 'n_runs': len(selected),
                     'positive_weeks': int(lengths.sum()), 'duration_mean': lengths.mean(),
                     'duration_min': lengths.min(), 'duration_p25': lengths.quantile(.25),
                     'duration_median': lengths.median(), 'duration_p75': lengths.quantile(.75),
                     'duration_max': lengths.max(), 'single_week_runs': int(lengths.eq(1).sum()),
                     'left_boundary_not_negative': int(selected.left_boundary.ne('negative_week').sum()),
                     'right_boundary_not_negative': int(selected.right_boundary.ne('negative_week').sum())})
    return pd.DataFrame(rows)


def validate_run_coverage(frame: pd.DataFrame, runs: pd.DataFrame) -> None:
    expected = {(row.municipality_code, row.week_start)
                for row in frame.loc[frame.label_eligible & frame.excess_label.eq(1)].itertuples()}
    observed = []
    for row in runs.itertuples():
        dates = pd.date_range(row.start_week, row.end_week, freq='7D')
        require(len(dates) == row.duration_weeks, 'Run endpoints disagree with its duration.')
        observed.extend((row.municipality_code, start) for start in dates)
    require(len(set(observed)) == len(observed) and set(observed) == expected,
            'Runs overlap, bridge nonpositive weeks or fail to cover positive labels.')


def build_tables(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    validate_label_frame(frame)
    by_municipality = ['municipality_code', 'municipality']
    annual = frame.loc[frame.week_start.dt.year.between(2012, 2025)].copy()
    annual['year'] = annual.week_start.dt.year
    runs = positive_runs(frame)
    validate_run_coverage(frame, runs)
    distribution = runs.groupby(by_municipality + ['duration_weeks'], sort=True).size().rename('n_runs').reset_index()
    distribution['pct_runs'] = 100 * distribution.n_runs / distribution.groupby('municipality_code').n_runs.transform('sum')
    tables = {
        'labels_global': label_counts(frame, []),
        'labels_by_municipality': label_counts(frame, by_municipality),
        'labels_by_year': label_counts(annual, ['year']),
        'labels_by_municipality_year': label_counts(annual, by_municipality + ['year']),
        'statistics_global': descriptive_statistics(frame, []),
        'statistics_by_municipality': descriptive_statistics(frame, by_municipality),
        'positive_runs': runs,
        'run_summary_global': run_summary(frame, runs, []),
        'run_summary_by_municipality': run_summary(frame, runs, by_municipality),
        'run_duration_distribution': distribution,
    }
    global_row = tables['labels_global'].iloc[0]
    for name in ['labels_by_municipality', 'labels_by_year', 'labels_by_municipality_year']:
        for column in ['eligible_weeks', 'positive_weeks', 'negative_weeks']:
            require(int(tables[name][column].sum()) == int(global_row[column]), 'Aggregate label totals disagree.')
    require(int(runs.duration_weeks.sum()) == int(global_row.positive_weeks), 'Run total disagrees with positives.')
    return tables, {'rows': len(frame), 'eligible_weeks': int(global_row.eligible_weeks),
                    'ineligible_weeks': int(global_row.ineligible_weeks),
                    'positive_weeks': int(global_row.positive_weeks), 'negative_weeks': int(global_row.negative_weeks),
                    'n_positive_runs': len(runs), 'run_coverage_exact': True,
                    'null_labels_counted_as_negative': False}

"""Describe zeros in approved historical references, without changing saved labels."""

import numpy as np
import pandas as pd

from .characterization import prepare_labels, positive_runs, validate_run_coverage
from .config import COUNT
from .ewt import historical_references, history_index, week_ordinal
from .validation import require


def reference_frame(source, product) -> tuple[pd.DataFrame, dict]:
    full = prepare_labels(product)
    require(product.select(source.column_names).equals(source, check_metadata=False), 'Original pilot fields differ.')
    index = history_index(source)
    # Independent enumeration checks the production alignment, including 53 -> 52.
    annual = {}
    for key, group in full.groupby(['municipality_code', full.week_start.dt.year], sort=True):
        annual[key] = group.sort_values('week_start').to_dict('records')
    frame = full.loc[full.label_eligible].copy()
    zeros, fallbacks = [], 0
    for row in frame.itertuples():
        ordinal = week_ordinal(row.week_start.date())
        references = historical_references(index, row.municipality_code, row.week_start.year, ordinal)
        require(len(references) == 5, 'An eligible week lacks five historical references.')
        for year, reference in zip(range(row.week_start.year - 5, row.week_start.year), references):
            weeks = annual[(row.municipality_code, year)]
            position = 52 if ordinal == 53 and len(weeks) == 52 else ordinal
            require(len(weeks) >= position, 'Incomplete independent alignment.')
            expected = weeks[position - 1]
            require(reference.week_start == expected['week_start'].date()
                    and reference.count == expected[COUNT] and not reference.partial,
                    'Historical date/count disagrees with independent alignment.')
            fallbacks += ordinal == 53 and position == 52
        zeros.append(sum(reference.count == 0 for reference in references))
    frame['historical_zero_count'] = np.asarray(zeros, dtype='int8')
    return frame, {'eligible_weeks': len(frame), 'historical_references_checked': len(frame) * 5,
                   'week_53_fallback_references': fallbacks,
                   'eligible_weeks_with_zero': int(frame.historical_zero_count.gt(0).sum()),
                   'positive_weeks_with_zero': int(frame.loc[frame.historical_zero_count.gt(0), 'excess_label'].sum())}


def _stats(values: pd.Series, prefix: str) -> dict:
    return {prefix + '_mean': values.mean(), prefix + '_std': values.std(ddof=1),
            prefix + '_min': values.min(), prefix + '_p25': values.quantile(.25),
            prefix + '_median': values.median(), prefix + '_p75': values.quantile(.75), prefix + '_max': values.max()}


def zero_summary(frame: pd.DataFrame, by_municipality: bool = False) -> pd.DataFrame:
    groups = frame.groupby(['municipality_code', 'municipality'], sort=True) if by_municipality else [(('ALL', 'Todos'), frame)]
    rows = []
    for (code, name), group in groups:
        positive_total = int(group.excess_label.sum())
        for n_zeros in range(6):
            selected = group.loc[group.historical_zero_count.eq(n_zeros)]
            positive = int(selected.excess_label.sum())
            rows.append({'municipality_code': code, 'municipality': name, 'historical_zero_count': n_zeros,
                         'eligible_weeks': len(selected), 'pct_eligible_weeks': 100 * len(selected) / len(group),
                         'positive_weeks': positive, 'negative_weeks': len(selected) - positive,
                         'positive_pct_in_group': 100 * positive / len(selected) if len(selected) else np.nan,
                         'pct_all_positive_weeks': 100 * positive / positive_total if positive_total else np.nan,
                         **_stats(selected.ewt, 'ewt'), **_stats(selected[COUNT], 'observed')})
    return pd.DataFrame(rows)


def municipal_quartiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed descriptive ranks over eligible 2012-2025; not training features."""
    enriched, cutoffs = frame.copy(), []
    for (code, name), group in frame.groupby(['municipality_code', 'municipality'], sort=True):
        record = {'municipality_code': code, 'municipality': name, 'eligible_weeks': len(group)}
        for column, prefix in [('ewt', 'ewt'), (COUNT, 'observed')]:
            limits = group[column].quantile([.25, .5, .75]).to_numpy(dtype=float)
            record.update({prefix + '_p' + str(q): value for q, value in zip([25, 50, 75], limits)})
            # Ties go into the lower interval. Never split equal values by rank.
            enriched.loc[group.index, prefix + '_quartile'] = np.searchsorted(limits, group[column], side='left') + 1
        cutoffs.append(record)
    for column in ['ewt_quartile', 'observed_quartile']:
        enriched[column] = enriched[column].astype('int8')
    enriched['low_ewt_descriptive'] = enriched.ewt_quartile.eq(1)
    enriched['high_count_descriptive'] = enriched.observed_quartile.eq(4)
    return enriched, pd.DataFrame(cutoffs)


def quartile_grid(frame: pd.DataFrame, by_municipality: bool = False) -> pd.DataFrame:
    groups = frame.groupby(['municipality_code', 'municipality'], sort=True) if by_municipality else [(('ALL', 'Todos'), frame)]
    rows = []
    for (code, name), group in groups:
        for ewt_q in range(1, 5):
            for count_q in range(1, 5):
                selected = group.loc[group.ewt_quartile.eq(ewt_q) & group.observed_quartile.eq(count_q)]
                positives = int(selected.excess_label.sum())
                rows.append({'municipality_code': code, 'municipality': name,
                             'ewt_quartile': ewt_q, 'observed_quartile': count_q,
                             'eligible_weeks': len(selected), 'positive_weeks': positives,
                             'negative_weeks': len(selected) - positives,
                             'positive_pct_in_cell': 100 * positives / len(selected) if len(selected) else np.nan})
    return pd.DataFrame(rows)


def combination_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [(('ALL', 'Todos'), frame)] + list(frame.groupby(['municipality_code', 'municipality'], sort=True))
    rows = []
    for (code, name), group in groups:
        n_positive = int(group.excess_label.sum())
        for low in [False, True]:
            for high in [False, True]:
                selected = group.loc[group.low_ewt_descriptive.eq(low) & group.high_count_descriptive.eq(high)]
                positive = int(selected.excess_label.sum())
                rows.append({'municipality_code': code, 'municipality': name, 'ewt_le_municipal_p25': low,
                             'observed_gt_municipal_p75': high, 'eligible_weeks': len(selected),
                             'positive_weeks': positive, 'negative_weeks': len(selected) - positive,
                             'positive_pct_in_group': 100 * positive / len(selected) if len(selected) else np.nan,
                             'pct_all_positive_weeks': 100 * positive / n_positive if n_positive else np.nan})
    return pd.DataFrame(rows)


def long_run_details(frame: pd.DataFrame, runs: pd.DataFrame, cutoffs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, annual_records = [], []
    long_runs = runs.loc[runs.duration_weeks.ge(52)].sort_values(['duration_weeks', 'municipality_code', 'start_week'],
                                                              ascending=[False, True, True]).reset_index(drop=True)
    thresholds = cutoffs.set_index('municipality_code')
    for i, run in enumerate(long_runs.itertuples()):
        selected = frame.loc[frame.municipality_code.eq(run.municipality_code)
                             & frame.week_start.between(run.start_week, run.end_week)].sort_values('week_start')
        require(len(selected) == run.duration_weeks and selected.excess_label.eq(1).all()
                and selected.week_start.diff().dropna().eq(pd.Timedelta(days=7)).all(), 'Long-run coverage is inconsistent.')
        prefix = {'run_id': f'L{i + 1:02d}', 'municipality_code': run.municipality_code, 'municipality': run.municipality}
        records.append({**prefix, 'start_week': run.start_week, 'end_week': run.end_week,
                        'duration_weeks': run.duration_weeks, 'left_boundary': run.left_boundary, 'right_boundary': run.right_boundary,
                        **_stats(selected.ewt, 'ewt'), **_stats(selected[COUNT], 'observed'),
                        'weeks_with_historical_zero': int(selected.historical_zero_count.gt(0).sum()),
                        'pct_weeks_with_historical_zero': 100 * selected.historical_zero_count.gt(0).mean(),
                        'historical_zero_references': int(selected.historical_zero_count.sum()),
                        'max_historical_zeros_per_week': int(selected.historical_zero_count.max()),
                        'pct_ewt_le_municipal_p25': 100 * selected.low_ewt_descriptive.mean(),
                        'pct_observed_gt_municipal_p75': 100 * selected.high_count_descriptive.mean(),
                        'municipal_ewt_p25': thresholds.loc[run.municipality_code, 'ewt_p25'],
                        'municipal_observed_p75': thresholds.loc[run.municipality_code, 'observed_p75']})
        for year, group in selected.groupby(selected.week_start.dt.year):
            annual_records.append({**prefix, 'year': int(year), 'weeks': len(group),
                                   'ewt_min': group.ewt.min(), 'ewt_median': group.ewt.median(), 'ewt_max': group.ewt.max(),
                                   'observed_min': group[COUNT].min(), 'observed_median': group[COUNT].median(),
                                   'observed_max': group[COUNT].max(),
                                   'weeks_with_historical_zero': int(group.historical_zero_count.gt(0).sum()),
                                   'pct_ewt_le_municipal_p25': 100 * group.low_ewt_descriptive.mean(),
                                   'pct_observed_gt_municipal_p75': 100 * group.high_count_descriptive.mean()})
    if not records:
        return (pd.DataFrame(columns=['run_id', 'municipality_code', 'municipality', 'start_week', 'end_week',
                                      'duration_weeks', 'weeks_with_historical_zero']),
                pd.DataFrame(columns=['run_id', 'municipality_code', 'municipality', 'year', 'weeks']))
    return pd.DataFrame(records), pd.DataFrame(annual_records)


def build_diagnostics(source, product) -> tuple[dict, pd.DataFrame, dict]:
    frame, audit = reference_frame(source, product)
    full = prepare_labels(product)
    runs = positive_runs(full)
    validate_run_coverage(full, runs)
    enriched, cutoffs = municipal_quartiles(frame)
    long_runs, annual = long_run_details(enriched, runs, cutoffs)
    tables = {'zeros_global': zero_summary(frame), 'zeros_by_municipality': zero_summary(frame, True),
              'municipal_quartile_cutoffs': cutoffs, 'quartile_grid_global': quartile_grid(enriched),
              'quartile_grid_by_municipality': quartile_grid(enriched, True),
              'low_reference_high_count_combinations': combination_summary(enriched),
              'long_runs_detail': long_runs, 'long_runs_annual_profiles': annual}
    positive = int(frame.excess_label.sum())
    for name in ['zeros_global', 'zeros_by_municipality', 'quartile_grid_global', 'quartile_grid_by_municipality']:
        require(int(tables[name].eligible_weeks.sum()) == len(frame)
                and int(tables[name].positive_weeks.sum()) == positive, 'Diagnostic aggregates do not reconcile.')
    require(int(annual.weeks.sum()) == int(long_runs.duration_weeks.sum()), 'Long-run annual profiles do not reconcile.')
    audit.update({'positive_weeks': positive, 'negative_weeks': len(frame) - positive, 'positive_runs': len(runs),
                  'long_runs': len(long_runs), 'weeks_in_long_runs': int(long_runs.duration_weeks.sum()),
                  'long_run_weeks_with_zero': int(long_runs.weeks_with_historical_zero.sum()),
                  'ineligible_weeks_in_statistics': 0})
    return tables, enriched, audit

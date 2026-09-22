"""Regression tests for PR #3 review findings: evidence-gate Cartesian
contracts, alternate-seed comparisons, and result-derived reporting."""
import json
import pandas as pd
import pytest
import optimizer_ci as ci
import optimizer_study as study
import transfer_study as t


def _keys_frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def test_run_keys_require_full_cartesian_product():
    full = [tuple([ds, seed, m]) for ds in t.PRIMARY_DATASETS for seed in study.SEEDS for m in study.METHODS]
    assert len(full) == 6 * 3 * 7
    ci.require_exact_keys(_keys_frame(full, ['dataset', 'seed', 'method']), ['dataset', 'seed', 'method'],
                          ci.expected_run_keys(), 'per_case.csv')
    # A whole dataset/seed/method run missing must fail, even though every
    # method, seed and dataset still appears marginally.
    partial = [r for r in full if r != ('jigsaw', 1602, 'guarded_selected')]
    with pytest.raises(AssertionError):
        ci.require_exact_keys(_keys_frame(partial, ['dataset', 'seed', 'method']), ['dataset', 'seed', 'method'],
                              ci.expected_run_keys(), 'per_case.csv')
    # Duplicated rows must fail even though the count looks right.
    duplicated = partial + [('jigsaw', 1602, 'compiled_legacy')]
    assert len(duplicated) == len(full)
    with pytest.raises(AssertionError):
        ci.require_exact_keys(_keys_frame(duplicated, ['dataset', 'seed', 'method']), ['dataset', 'seed', 'method'],
                              ci.expected_run_keys(), 'per_case.csv')
    # An empty gate table must fail instead of passing as a no-op loop.
    with pytest.raises(AssertionError):
        ci.require_exact_keys(_keys_frame([], ['dataset', 'seed'])[0:0], ['dataset', 'seed'],
                              ci.expected_gate_keys(), 'promotion_gates.csv')


def _comparison_frame(seed):
    rows = []
    for dataset in t.PRIMARY_DATASETS:
        for method in ('legacy_bellman', 'polarity_bellman', 'context_polarity', 'tuning_selected', 'guarded_selected'):
            for group in range(3):
                for sample in range(2):
                    rows.append(dict(dataset=dataset, seed=seed, method=method,
                                     sample_id=f'{dataset}-{group}-{sample}', group=f'{dataset}-g{group}',
                                     objective=0.2 + 0.01 * sample))
    return pd.DataFrame(rows)


def test_comparisons_supports_alternate_primary_seed():
    frame = _comparison_frame(1602)
    out = study.comparisons(frame, 1602)
    assert set(out.method) == {'polarity_bellman', 'context_polarity', 'tuning_selected', 'guarded_selected'}
    # The default primary seed has no rows here, so it must fail loudly
    # instead of bootstrapping an empty sample.
    with pytest.raises(ValueError):
        study.comparisons(frame)


def _write_report_inputs(root, toxic_n=20, nontoxic_n=300, promote=False):
    rdir = root / 'results' / 'optimizer'
    rdir.mkdir(parents=True, exist_ok=True)
    (root / 'verification').mkdir(exist_ok=True)
    (root / 'verification' / 'optimizer.json').write_text(json.dumps({
        'all_gates_passed': True, 'tested_commit': 'test-commit', 'run_url': 'http://example',
        'tests': 10, 'ledger_rows': 100, 'exported_policies': 126, 'full_pattern_equivalence_cases': 936}))
    methods = ['legacy_bellman', 'compiled_legacy', 'polarity_bellman', 'context_bellman',
               'context_polarity', 'tuning_selected', 'guarded_selected']
    pd.DataFrame([dict(seed=1601, method=m, objective=0.15, coverage=0.8, queries=1.0) for m in methods]
                 ).to_csv(rdir / 'macro.csv', index=False)
    gates = [dict(dataset=ds, seed=seed, promote=False) for ds in t.PRIMARY_DATASETS for seed in study.SEEDS]
    if promote:
        gates[0]['promote'] = True
        for row in gates:
            if row['dataset'] == 'jigsaw' and row['seed'] == 1601:
                row['promote'] = True
    pd.DataFrame(gates).to_csv(rdir / 'promotion_gates.csv', index=False)
    pd.DataFrame([dict(dataset='macro_six', method=m, difference=-0.01, lo=-0.02, hi=0.0,
                        family_lo=-0.03, family_hi=0.01)
                    for m in ('polarity_bellman', 'context_polarity', 'tuning_selected', 'guarded_selected')]
                   ).to_csv(rdir / 'comparisons.csv', index=False)
    pd.DataFrame([dict(seed=1601, dataset=ds, method=m, objective=0.15)
                    for ds in t.PRIMARY_DATASETS
                    for m in ('legacy_bellman', 'polarity_bellman', 'context_polarity')]
                   ).to_csv(rdir / 'summary.csv', index=False)
    pd.DataFrame([dict(dataset=ds, implementation=impl, median_ns=100.0)
                    for ds in t.PRIMARY_DATASETS for impl in ('legacy', 'compiled')]
                   ).to_csv(rdir / 'runtime.csv', index=False)
    pd.DataFrame([
        dict(seed=1601, dataset='jigsaw', method='legacy_bellman', gold='yes', n=toxic_n,
             correct=0, wrong=2, deferred=toxic_n - 2),
        dict(seed=1601, dataset='jigsaw', method='legacy_bellman', gold='no', n=nontoxic_n,
             correct=nontoxic_n - 30, wrong=30, deferred=0),
        dict(seed=1601, dataset='jigsaw', method='polarity_bellman', gold='yes', n=toxic_n,
             correct=1, wrong=1, deferred=toxic_n - 2),
        dict(seed=1601, dataset='jigsaw', method='polarity_bellman', gold='no', n=nontoxic_n,
             correct=nontoxic_n - 5, wrong=5, deferred=0),
        dict(seed=1601, dataset='jigsaw', method='context_polarity', gold='yes', n=toxic_n,
             correct=2, wrong=0, deferred=toxic_n - 2),
        dict(seed=1601, dataset='jigsaw', method='context_polarity', gold='no', n=nontoxic_n,
             correct=nontoxic_n - 8, wrong=8, deferred=0),
    ]).to_csv(rdir / 'class_outcomes.csv', index=False)
    pd.DataFrame([dict(dataset='jigsaw', seed=1601, fit_seconds=1.0, train_questions=10,
                        representation_questions=4, risk_calibration_questions=4, tuning_questions=2,
                        gate_questions=2, fitting_archive_responses=100)]).to_csv(rdir / 'fit_resources.csv', index=False)


def test_report_derives_jigsaw_counts_and_gate_status(tmp_path, monkeypatch):
    import optimizer_report as rep
    _write_report_inputs(tmp_path, toxic_n=20, nontoxic_n=300, promote=False)
    monkeypatch.chdir(tmp_path)
    rep.main()
    en = (tmp_path / 'dist-optimizer' / 'report.en.md').read_text()
    zh = (tmp_path / 'dist-optimizer' / 'report.zh-TW.md').read_text()
    assert '300 non-toxic cases' in en and '20 positive cases' in en
    assert '290' not in en and 'among 14 positive' not in en
    assert 'remains the incumbent on the reported seed' in en
    assert '300個非毒性' in zh and '20個毒性' in zh
    _write_report_inputs(tmp_path, toxic_n=20, nontoxic_n=300, promote=True)
    rep.main()
    en = (tmp_path / 'dist-optimizer' / 'report.en.md').read_text()
    assert 'promotes a challenger on the reported seed' in en

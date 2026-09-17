"""Regression tests for the publication gates, not fabricated benchmark results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import ci_external_evidence as c


def outputs(path: Path):
    path.mkdir()
    for name in c.REQUIRED:
        file = path/name
        if name.endswith('.npz'):
            np.savez_compressed(file, answers=np.array([[0, 1, 4, 2]], dtype=np.int8))
        elif name.endswith('.json'):
            file.write_text(json.dumps({'stable': 1, 'elapsed_seconds': 1}))
        else:
            file.write_text('id,value\n0,1\n')
    return path


def test_all_required_outputs_are_checked(tmp_path):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    result = c.compare_runs(a, b)
    assert len(result) == 21
    assert all(row['byte_identical'] for row in result)


def test_only_provenance_runtime_fields_may_differ(tmp_path):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    (b/'provenance.json').write_text(json.dumps({'stable': 1, 'elapsed_seconds': 8, 'python': 'different'}))
    assert len(c.compare_runs(a, b)) == 21


@pytest.mark.parametrize('name', ['primary_per_case.csv', 'mmlu_gold_audit.json', 'primary_summary.csv'])
def test_changed_scientific_result_fails(tmp_path, name):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    (b/name).write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        c.compare_runs(a, b)


def test_changed_nonvolatile_provenance_fails(tmp_path):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    (b/'provenance.json').write_text(json.dumps({'stable': 2}))
    with pytest.raises(ValueError, match='provenance differs'):
        c.compare_runs(a, b)


@pytest.mark.parametrize('side', ['first', 'second'])
def test_missing_file_fails(tmp_path, side):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    ((a if side == 'first' else b)/'primary_summary.csv').unlink()
    with pytest.raises(ValueError):
        c.compare_runs(a, b)


def test_extra_file_fails(tmp_path):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    (b/'unexpected.csv').write_text('bad')
    with pytest.raises(ValueError, match='sets differ'):
        c.compare_runs(a, b)


def test_array_tampering_fails(tmp_path):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    np.savez_compressed(b/'derived_traces.npz', answers=np.array([[0, 0, 4, 2]], dtype=np.int8))
    with pytest.raises(AssertionError):
        c.compare_runs(a, b)


def test_pickle_array_is_rejected(tmp_path):
    a, b = outputs(tmp_path/'a'), outputs(tmp_path/'b')
    np.savez_compressed(b/'derived_traces.npz', answers=np.array([{'bad': 1}], dtype=object))
    with pytest.raises(ValueError):
        c.compare_runs(a, b)


@pytest.mark.parametrize('element', ['<failure/>', '<error/>', '<skipped/>'])
def test_unsuccessful_junit_fails(tmp_path, element):
    p = tmp_path/'tests.xml'
    p.write_text(f'<testsuites><testsuite><testcase>{element}</testcase></testsuite></testsuites>')
    with pytest.raises(ValueError, match='Tests empty, failing or skipped'):
        c.test_counts(p)


def test_empty_junit_fails(tmp_path):
    p = tmp_path/'tests.xml'; p.write_text('<testsuites/>')
    with pytest.raises(ValueError):
        c.test_counts(p)


def test_junit_counts_actual_testcases(tmp_path):
    p = tmp_path/'tests.xml'
    p.write_text('<testsuites tests="999"><testsuite tests="999"><testcase/><testcase/></testsuite></testsuites>')
    assert c.test_counts(p) == {'tests': 2, 'failures': 0, 'skipped': 0}


def test_empty_primary_fails(tmp_path):
    pd.DataFrame(columns=['policy']).to_csv(tmp_path/'primary_per_case.csv', index=False)
    with pytest.raises(ValueError, match='Empty primary'):
        c.validate_primary(tmp_path)


def test_missing_policy_fails(tmp_path):
    pd.DataFrame({'policy': ['bellman']}).to_csv(tmp_path/'primary_per_case.csv', index=False)
    with pytest.raises(ValueError, match='policies'):
        c.validate_primary(tmp_path)

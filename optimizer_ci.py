"""Evidence gate: validate new decisions without requiring a benchmark victory."""
from __future__ import annotations
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import adaptive_router as ar
import optimizer_study as study
import transfer_study as t


def expected_run_keys():
    return {(ds, int(seed), m) for ds in t.PRIMARY_DATASETS for seed in study.SEEDS for m in study.METHODS}


def expected_gate_keys():
    return {(ds, int(seed)) for ds in t.PRIMARY_DATASETS for seed in study.SEEDS}


def observed_keys(frame, columns):
    keys = set()
    for row in frame.itertuples():
        key = []
        for column in columns:
            value = getattr(row, column)
            key.append(int(value) if column == 'seed' else str(value))
        keys.add(tuple(key))
    return keys


def require_exact_keys(frame, columns, expected, what):
    observed = observed_keys(frame, columns)
    if observed != expected or len(frame) != len(expected):
        raise AssertionError(f'{what} must hold exactly one row per key: {len(observed)}/{len(expected)} distinct')


def test_count(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    if len(cases)<377:raise AssertionError('Incomplete regression suite')
    if any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):
        raise AssertionError('Failed or skipped regression tests')
    return len(cases)


def validate(directory,cache):
    ledger=pd.read_csv(directory/'per_case.csv',dtype={'sample_id':str})
    manifest=pd.read_csv(directory/'split_roles.csv',dtype={'sample_id':str})
    expected_runs = expected_run_keys()
    require_exact_keys(ledger, ['dataset', 'seed', 'method'], expected_runs, 'per_case.csv')
    assert not ledger.duplicated(['dataset','sample_id','seed','method']).any()
    assert np.isfinite(ledger[['objective','loss','queries','cost']]).all().all()
    assert ledger.queries.between(0,3).all() and (ledger.queries==ledger.queries.astype(int)).all()
    np.testing.assert_allclose(ledger.objective,ledger.loss+ledger.cost,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ledger.cost,ledger.queries*.01,rtol=0,atol=1e-12)
    assert ((ledger.selected==-1)==(ledger.answered==0)).all()
    assert (ledger[ledger.answered==0].loss==.25).all()
    for seed,g in manifest.groupby('seed'):
        assert (g.groupby('group').role.nunique()==1).all()
        assert set(g.role)=={'representation','risk_calibration','tuning','promotion_gate','test'}
    source=json.loads((t.ROOT/'transfer_protocol.json').read_text())
    for ds in t.PRIMARY_DATASETS:
        data,_=t.load_dataset(cache/f'{ds}_predictions.csv',ds,source['source_sha256'][ds])
        expected=set(data.ids[data.splits=='test'].astype(str))
        errors=dict(zip(data.ids.astype(str),data.errors['replacement']))
        for (seed,method),g in ledger[ledger.dataset==ds].groupby(['seed','method']):
            assert set(g.sample_id)==expected
            for r in g.itertuples():
                order=() if pd.isna(r.query_order) else tuple(int(x) for x in str(r.query_order).split(','))
                assert len(order)==r.queries and len(set(order))==len(order)
                assert all(i in range(4) for i in order) and (r.selected==-1 or r.selected in order)
                assert r.loss==(.25 if r.selected<0 else errors[r.sample_id][r.selected])
    generated=study.summarize(ledger).sort_values(['dataset','seed','method']).reset_index(drop=True)
    stated=pd.read_csv(directory/'summary.csv').sort_values(['dataset','seed','method']).reset_index(drop=True)
    pd.testing.assert_frame_equal(generated,stated,check_dtype=False,atol=1e-12,rtol=1e-10)
    checks=pd.read_csv(directory/'compilation_checks.csv')
    assert len(checks)==18 and (checks.complete_patterns==52).all() and checks.exact_action_choice_equivalence.all()
    for seed in study.SEEDS:
        old=ledger[(ledger.seed==seed)&(ledger.method=='legacy_bellman')].sort_values(['dataset','sample_id'])
        new=ledger[(ledger.seed==seed)&(ledger.method=='compiled_legacy')].sort_values(['dataset','sample_id'])
        cols=['selected','query_order','objective']
        pd.testing.assert_frame_equal(old[cols].reset_index(drop=True),new[cols].reset_index(drop=True),check_dtype=False)
    gates=pd.read_csv(directory/'promotion_gates.csv');gatedata=pd.read_csv(directory/'gate_cases.csv')
    require_exact_keys(gates, ['dataset', 'seed'], expected_gate_keys(), 'promotion_gates.csv')
    for row in gates.itertuples():
        g=gatedata[(gatedata.dataset==row.dataset)&(gatedata.seed==row.seed)]
        eligible=manifest[(manifest.dataset==row.dataset)&(manifest.seed==row.seed)&(manifest.role=='promotion_gate')]
        assert set(g.group)==set(eligible.group) and not g.group.duplicated().any()
        bound=ar.promotion_bound((g.candidate_risk-g.incumbent_risk).to_numpy())
        assert abs(bound['upper_difference']-row.upper_difference)<1e-10
        assert bound['promote']==row.promote
        f=ledger[(ledger.dataset==row.dataset)&(ledger.seed==row.seed)]
        x=f[f.method=='guarded_selected'].sort_values('sample_id')
        if not row.promote:
            y=f[f.method=='compiled_legacy'].sort_values('sample_id')
            pd.testing.assert_frame_equal(x[['selected','objective','query_order']].reset_index(drop=True),y[['selected','objective','query_order']].reset_index(drop=True))
        else:
            y=f[f.method=='tuning_selected'].sort_values('sample_id')
            pd.testing.assert_frame_equal(x[['selected','objective','query_order']].reset_index(drop=True),y[['selected','objective','query_order']].reset_index(drop=True))
    exports=pd.read_csv(directory/'exports.csv')
    require_exact_keys(exports, ['dataset', 'seed', 'method'], expected_runs, 'exports.csv')
    assert exports.roundtrip_exact_decisions.all()
    for r in exports.itertuples():
        path=directory/'policies'/f'{r.dataset}-{r.seed}-{r.method}.json'
        assert t.digest(path)==r.sha256 and path.stat().st_size==r.bytes
    return len(ledger),len(exports)


def compare(first,second):
    names={str(p.relative_to(first)) for p in first.rglob('*') if p.is_file()}
    if names!={str(p.relative_to(second)) for p in second.rglob('*') if p.is_file()}:raise AssertionError('File inventories differ')
    checks=[]
    for name in sorted(names):
        p,q=first/name,second/name
        if name in ('runtime.csv','fit_resources.csv'):
            x,y=pd.read_csv(p),pd.read_csv(q)
            columns=['median_ns','p95_ns'] if name=='runtime.csv' else ['fit_seconds']
            assert np.isfinite(x[columns]).all().all() and (x[columns]>=0).all().all()
            assert np.isfinite(y[columns]).all().all() and (y[columns]>=0).all().all()
            pd.testing.assert_frame_equal(x.drop(columns=columns),y.drop(columns=columns))
            checks.append(dict(file=name,kind='non-timing columns equal',byte_identical=False));continue
        if name.endswith('.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(q),check_dtype=False,atol=1e-12,rtol=1e-10)
        else:
            assert p.read_bytes()==q.read_bytes(),name
        checks.append(dict(file=name,kind='scientific',byte_identical=t.digest(p)==t.digest(q),sha256=t.digest(p)))
    return checks


def run(first,second,cache,xml,out):
    tests=test_count(xml);n,policies=validate(first,cache);validate(second,cache)
    checks=compare(first,second)
    protocol=json.loads((t.ROOT/'optimizer_protocol.json').read_text())
    rawprovenance=json.loads((first/'provenance.json').read_text())
    assert rawprovenance['protocol_sha256']==t.digest(t.ROOT/'optimizer_protocol.json')
    assert rawprovenance['new_llm_calls']==0 and rawprovenance['independent_new_test_questions']==0
    result=dict(tested_commit=os.getenv('GITHUB_SHA','local'),tests=tests,failures=0,skipped=0,ledger_rows=n,
                exported_policies=policies,all_gates_passed=True,complete_process_runs=2,comparisons=checks,
                full_pattern_equivalence_cases=6*3*52,protocol_sha256=rawprovenance['protocol_sha256'],
                source=rawprovenance,run_url=os.getenv('RESEARCH_RUN_URL'))
    out.parent.mkdir(parents=True,exist_ok=True);t.write_json(out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','source')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',type=Path,default=Path('results/optimizer'));p.add_argument('--second',type=Path,default=Path('results/optimizer-repeat'))
    p.add_argument('--cache',type=Path,default=t.ROOT/'.cache/proeval');p.add_argument('--xml',type=Path,default=Path('verification/optimizer-tests.xml'))
    p.add_argument('--out',type=Path,default=Path('verification/optimizer.json'));a=p.parse_args();run(a.first,a.second,a.cache,a.xml,a.out)

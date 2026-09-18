"""Fail-closed two-process evidence validation for v1.3. No winner assertion."""
import argparse,hashlib,json,os,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import transfer_study as t

REQUIRED=('per_case.csv','summary.csv','paired_comparisons.csv','selection.csv','cost_sweep.csv','refresh_costs.csv','source_audit.json','provenance.json','split_manifest.csv','controller_timing.csv','calibration_resources.csv','numeric_tolerance_audit.csv','numeric_disagreements.csv','label_range_audit.csv','audit_scope.json')

def check_xml(path):
    root=ET.parse(path).getroot();cases=list(root.iter('testcase'))
    if len(cases)<200:raise AssertionError('Missing test coverage')
    if any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):raise AssertionError('Failed or skipped tests')
    return len(cases)

def check_ledgers(path):
    f=pd.read_csv(path/'per_case.csv');s=pd.read_csv(path/'summary.csv');manifest=pd.read_csv(path/'split_manifest.csv')
    if f.empty or f.duplicated(['dataset','sample_id','phase','policy','price']).any():raise AssertionError('Empty/duplicate ledger')
    assert np.isfinite(f[['loss','cost','objective','queries']]).all().all()
    assert ((f.queries>=0)&(f.queries<=3)&(f.queries==f.queries.astype(int))).all()
    np.testing.assert_allclose(f.cost,f.price*f.queries,rtol=0,atol=1e-12)
    np.testing.assert_allclose(f.objective,f.loss+f.cost,rtol=0,atol=1e-12)
    assert ((f.answered==0)==(f.selected==-1)).all()
    assert ((f.loc[f.answered==0,'loss']-f.loc[f.answered==0,'defer_loss']).abs()<1e-12).all()
    for r in f.itertuples():
        used=[] if pd.isna(r.query_order) else list(map(int,str(r.query_order).split(',')))
        if len(used)!=r.queries or len(used)!=len(set(used)) or any(i not in range(4) for i in used):raise AssertionError('Invalid query ledger')
        if r.selected>=0 and r.selected not in used:raise AssertionError('Unacquired candidate')
    m=manifest[manifest['split']=='test']
    for (dataset,phase),g in f.groupby(['dataset','phase']):
        expected=set(m.loc[m.dataset==dataset,'id'].astype(str))
        policies=set(t.MODES) if not phase.startswith('refresh_') else {'single','static','myopic','bellman'}
        assert set(g.policy)==policies
        for _,h in g.groupby('policy'):assert set(h.sample_id.astype(str))==expected
    assert (manifest.groupby('group')['split'].nunique()==1).all()
    calculated=t.summarize(f)
    cols=['dataset','phase','policy','price']
    s=s.sort_values(cols).reset_index(drop=True);calculated=calculated.sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(s,calculated,check_dtype=False,rtol=1e-9,atol=1e-12)
    ordinal=s[s.dataset=='dices'];assert ordinal.brier.isna().all() and ordinal.ece.isna().all()
    return len(f)

def verify(first,second,xml,out):
    count=check_xml(xml);case_count=check_ledgers(first);check_ledgers(second)
    for name in REQUIRED:
        if not (first/name).is_file() or not (second/name).is_file():raise AssertionError('Required output missing: '+name)
    names={p.name for p in first.iterdir() if p.is_file()}
    if names!={p.name for p in second.iterdir() if p.is_file()}:raise AssertionError('Output sets differ')
    checks=[];skipped_runtime=[]
    for name in sorted(names):
        a,b=first/name,second/name
        if name=='controller_timing.csv':
            x=pd.read_csv(a).drop(columns='seconds');y=pd.read_csv(b).drop(columns='seconds')
            pd.testing.assert_frame_equal(x,y);skipped_runtime.append(name);continue
        if name=='calibration_resources.csv':
            x=pd.read_csv(a).drop(columns='train_select_seconds');y=pd.read_csv(b).drop(columns='train_select_seconds')
            pd.testing.assert_frame_equal(x,y);skipped_runtime.append(name);continue
        if name.endswith('.npz'):
            with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
                assert x.files==y.files
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        elif name.endswith('.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(a),pd.read_csv(b),check_exact=False,rtol=1e-10,atol=1e-12)
        else:
            if a.read_bytes()!=b.read_bytes():raise AssertionError('Nondeterministic output: '+name)
        checks.append({'file':name,'sha256_first':t.digest(a),'sha256_second':t.digest(b),'byte_identical':t.digest(a)==t.digest(b)})
    protocol=t.digest(t.ROOT/'transfer_protocol.json')
    if protocol!='ee7945c146b99a3aad5072db30aca208e7d0d0979060927826f4ea6b8d14032a':raise AssertionError('Locked protocol changed')
    provenance=json.loads((first/'provenance.json').read_text())
    assert provenance['protocol_sha256']==protocol and provenance['new_llm_api_calls']==provenance['jev_calls']==provenance['cera_agent_training_steps']==0
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit='local-uncommitted-worktree'
    record={'version':'1.3.0','tested_commit':commit,'tests':count,'failures':0,'skipped':0,'ledger_rows':case_count,
        'separate_process_runs':2,'comparisons':checks,'runtime_only_columns_excluded':skipped_runtime,
        'protocol_sha256':protocol,'all_gates_passed':True,'source':provenance,
        'run_url':os.getenv('RESEARCH_RUN_URL'),'limitations':'Same-code deterministic rerun, not independent scientific replication. Archived model-pool replacement is not within-agent continual training. Jev is not run.'}
    out.parent.mkdir(parents=True,exist_ok=True);t.write_json(out,record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('comparisons','source')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',type=Path,default=Path('results/transfer'));p.add_argument('--second',type=Path,default=Path('results/transfer-repeat'));p.add_argument('--xml',type=Path,default=Path('verification/transfer-tests.xml'));p.add_argument('--out',type=Path,default=Path('verification/transfer.json'));a=p.parse_args();verify(a.first,a.second,a.xml,a.out)

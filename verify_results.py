"""Fail closed on changed numbers, missing tests, or non-reproducible arrays."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def compare(a,b):
    if a.suffix=='.csv':
        x,y=pd.read_csv(a),pd.read_csv(b)
        pd.testing.assert_frame_equal(x,y,check_exact=False,rtol=1e-11,atol=1e-12)
    elif a.suffix=='.npz':
        with np.load(a) as x,np.load(b) as y:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    elif a.suffix=='.json': assert json.loads(a.read_text())==json.loads(b.read_text())
    else: assert a.read_bytes()==b.read_bytes()

def main(repeat,final):
    rows=[]
    for b in sorted(repeat.rglob('*')):
        if not b.is_file() or b.name=='run_metadata.json':continue
        a=ROOT/'results'/b.relative_to(repeat)
        assert a.is_file(),f'Missing result {a}'
        compare(a,b)
        rows.append({'file':str(a.relative_to(ROOT)), 'byte_identical':a.read_bytes()==b.read_bytes(),
                     'sha256':sha(a),'repeat_sha256':sha(b)})
    assert len(rows)==34, f'Expected 34 main outputs, found {len(rows)}'
    for f in (ROOT/'expected').glob('*.csv'):compare(f,ROOT/'results'/f.name)
    tree=ET.parse(ROOT/'verification/junit.xml')
    cases=tree.findall('.//testcase')
    assert len(cases)==86,f'Expected 86 tests, found {len(cases)}'
    assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    if final:
        m=pd.read_csv(ROOT/'results/revision/matched_summary.csv').set_index('policy')
        for k,v in {'R0_stop':.978159,'R1_static_all':.38170844,'R2_greedy_workers':.894874,'R3_dp_workers':.848145}.items():
            assert abs(m.loc[k,'objective']-v)<1e-6,(k,m.loc[k,'objective'],v)
        learning=pd.read_csv(ROOT/'results/revision/calibration_curve.csv')
        assert len(learning)==300 and learning.objective.notna().all()
        for name in ['matched_comparisons.csv','routing_noise.csv','verifier_coupling.csv','preference_information_bound.csv']:
            assert (ROOT/'results/revision'/name).is_file()
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL,text=True).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):commit=None
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.iterdir()) if p.suffix in {'.py','.json','.sh','.md','.tex','.txt'} and p.is_file() and not p.name.startswith('paper.')}
    result={'version':'1.1.0','tested_commit':commit,'python':sys.version,'platform':platform.platform(),
            'tests':len(cases),'failures':0,'skipped':0,'main_comparison_count':len(rows),
            'all_numerical_comparisons_passed':True,'byte_identical_main_count':sum(r['byte_identical'] for r in rows),
            'reference_tables_passed':True,'revision_complete':final,'llm_api_calls':0,
            'comparisons':rows,'source_sha256':sources,
            'limitations':'Same implementation/generator reproduction, not external scientific replication or real LLM evaluation.'}
    (ROOT/'verification/attestation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','source_sha256')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeat',type=Path,required=True);p.add_argument('--final',action='store_true')
    args=p.parse_args();main(args.repeat,args.final)

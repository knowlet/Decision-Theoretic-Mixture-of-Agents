"""Paired head-to-head analysis and integrity gates. Never asserts our method wins."""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path
import numpy as np
import pandas as pd
import openjev_benchmark as b
import transfer_study as t

def validate_bundle(path):
    checks=json.loads((path/'SHA256.json').read_text())
    for name,sha in checks.items():
        if Path(name).name!=name or t.digest(path/name)!=sha:raise AssertionError('Artifact hash mismatch')
    meta=json.loads((path/'provenance.json').read_text());manifest=json.loads((path/'case_manifest.json').read_text())
    if meta['upstream_direct_git_blob']!='943f34728d1966bf2325d6e6d34d2a7043cd6a56':raise AssertionError('Scorer provenance mismatch')
    if meta['request_spec_sha256']!=t.digest(t.ROOT/'openjev_protocol.json'):raise AssertionError('Protocol mismatch')
    assert len(manifest)==192 and len({r['id'] for r in manifest})==192
    test={r['id']:r for r in manifest if r['split']=='test'};dev={r['id']:r for r in manifest if r['split']=='dev'}
    assert len(test)==128 and len(dev)==64
    assert not {r['group'] for r in test.values()}&{r['group'] for r in dev.values()}
    scores=[json.loads(x) for x in (path/'selector_scores.jsonl').read_text().splitlines()]
    assert len(scores)==192 and {r['id'] for r in scores}=={r['id'] for r in manifest}
    for r in scores:
        p=np.array(r['probabilities']);assert len(p)==4 and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-6
        assert r['model']['source']==meta['metadata']['source'] and r['model']['revision']==meta['metadata']['revision']
    df=pd.read_csv(path/'per_case.csv',dtype={'sample_id':str})
    assert df.algorithm.nunique()==12 and len(df)==12*128 and not df.duplicated(['dataset','sample_id','algorithm']).any()
    np.testing.assert_allclose(df.objective,df.loss+df.cost,atol=1e-12,rtol=0)
    np.testing.assert_allclose(df.cost,df.queries*.01,atol=1e-12,rtol=0)
    assert np.isfinite(df[['objective','loss','queries']]).all().all() and df.queries.between(0,3).all()
    for alg,g in df.groupby('algorithm'):
        assert set(g.dataset+':'+g.sample_id)==set(test)
        if alg in ('empirical_fixed3','matched_majority3','openjev_raw_fixed3','openjev_calibrated_fixed3'):
            assert (g.queries==3).all()
            for r in g.itertuples():assert r.selected==-1 or r.selected in test[r.dataset+':'+r.sample_id]['panel']
        if alg.startswith('openjev_'):assert (g.new_selector_calls==1).all()
        else:assert (g.new_selector_calls==0).all()
    assert ((df.selected==-1)==(df.answered==0)).all()
    assert (df.loc[df.answered==0,'loss']==.25).all()
    rep=json.loads((path/'repeat_check.json').read_text());assert len(rep)==8 and all(r['same_argmax'] and r['max_probability_delta']<1e-5 for r in rep)
    assert len(pd.read_csv(path/'order_robustness.csv'))==32
    assert meta['new_primary_local_forwards']==192 and meta['new_robustness_local_forwards']==40 and meta['warmup_forwards']==1
    assert meta['new_llm_api_calls']==meta['jev_proprietary_calls']==meta['cera_training_steps']==meta['worker_generation_calls']==0
    return df,meta,manifest

def summarize(df):
    rows=[]
    for keys,g in df.groupby(['selector_model','algorithm','dataset']):
        m,a,d=keys;ans=g[g.answered==1]
        rows.append(dict(model=m,algorithm=a,dataset=d,n=len(g),objective=g.objective.mean(),coverage=g.answered.mean(),
            accuracy_on_answered=1-ans.loss.mean(),wrong_per_all=((g.answered==1)&(g.loss==1)).mean(),
            queries=g.queries.mean(),controller_seconds_median=g.controller_seconds.median(),controller_seconds_p95=g.controller_seconds.quantile(.95),input_tokens_mean=g.input_tokens.mean()))
    result=pd.DataFrame(rows)
    macro=result.groupby(['model','algorithm'],sort=True).agg(n=('n','sum'),objective=('objective','mean'),coverage=('coverage','mean'),queries=('queries','mean'),wrong_per_all=('wrong_per_all','mean'),input_tokens_mean=('input_tokens_mean','mean')).reset_index()
    for i,r in macro.iterrows():
        g=df[(df.selector_model==r.model)&(df.algorithm==r.algorithm)];ans=g[g.answered==1]
        macro.loc[i,'accuracy_on_answered']=1-ans.loss.mean();macro.loc[i,'controller_seconds_median']=g.controller_seconds.median();macro.loc[i,'controller_seconds_p95']=g.controller_seconds.quantile(.95)
    return result,macro

def paired(df,count=2000):
    contrasts=[('qwen35-4b','openjev_raw_fixed3','qwen35-4b','empirical_fixed3'),('qwen35-4b','openjev_calibrated_fixed3','qwen35-4b','empirical_fixed3'),('qwen35-4b','openjev_calibrated_fixed3','qwen35-4b','empirical_bellman'),('qwen35-0.8b','openjev_raw_fixed3','qwen35-4b','openjev_raw_fixed3')]
    rows=[]
    for ma,aa,mb,ab in contrasts:
        diffs=[];draws=[]
        for i,ds in enumerate(b.DATASETS):
            x=df[(df.selector_model==ma)&(df.algorithm==aa)&(df.dataset==ds)].set_index('sample_id');y=df[(df.selector_model==mb)&(df.algorithm==ab)&(df.dataset==ds)].set_index('sample_id').reindex(x.index)
            assert len(x)==32 and not y.objective.isna().any()
            d=(x.objective-y.objective).to_numpy();rng=np.random.default_rng(np.random.SeedSequence([141809,i]));boot=d[rng.integers(0,len(d),(count,len(d)))].mean(1)
            diffs.append(d.mean());draws.append(boot);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(dataset=ds,a=ma+':'+aa,b=mb+':'+ab,difference=d.mean(),lo=lo,hi=hi,n=32))
        lo,hi=np.quantile(np.mean(draws,axis=0),[.025,.975]);flo,fhi=np.quantile(np.mean(draws,axis=0),[.00625,.99375]);rows.append(dict(dataset='macro_binary',a=ma+':'+aa,b=mb+':'+ab,difference=np.mean(diffs),lo=lo,hi=hi,family_lo=flo,family_hi=fhi,n=128))
    return pd.DataFrame(rows)

def calibration(path,model):
    f=pd.read_csv(path/'confidence.csv');rows=[]
    for score in ('raw_option_score','calibrated_success'):
        p=f[score].to_numpy();y=f.correct.to_numpy();bins=np.minimum((p*10).astype(int),9);ece=sum(np.mean(bins==i)*abs(p[bins==i].mean()-y[bins==i].mean()) for i in range(10) if np.any(bins==i))
        rows.append(dict(model=model,score=score,n=len(f),brier=np.mean((p-y)**2),ece=ece,caveat='Raw action-option mass is only a confidence proxy, not a correctness distribution; calibrated variant uses 64 distinct development cases.'))
    return rows

def run(inputs,out):
    out.mkdir(parents=True,exist_ok=True);frames=[];metas=[];manifests=[];cal=[];robust=[]
    for label in b.MODELS:
        path=inputs/label;f,m,manifest=validate_bundle(path);frames.append(f);metas.append(m);manifests.append(manifest);cal.extend(calibration(path,label))
        r=pd.read_csv(path/'order_robustness.csv');robust.append(dict(model=label,n=len(r),flips=int(r.flipped.sum()),mean_tv=r.tv.mean()))
    assert manifests[0]==manifests[1]
    a=frames[0][~frames[0].algorithm.str.startswith('openjev_')].sort_values(['dataset','sample_id','algorithm']).reset_index(drop=True);c=frames[1][~frames[1].algorithm.str.startswith('openjev_')].sort_values(['dataset','sample_id','algorithm']).reset_index(drop=True)
    cols=['dataset','sample_id','algorithm','selected','answered','loss','queries','objective'];pd.testing.assert_frame_equal(a[cols],c[cols],check_dtype=False,rtol=1e-12,atol=1e-12)
    df=pd.concat(frames,ignore_index=True);bytask,macro=summarize(df);comp=paired(df)
    df.to_csv(out/'paired_cases.csv',index=False);bytask.to_csv(out/'by_dataset.csv',index=False);macro.to_csv(out/'summary.csv',index=False);comp.to_csv(out/'comparisons.csv',index=False);pd.DataFrame(cal).to_csv(out/'calibration.csv',index=False);pd.DataFrame(robust).to_csv(out/'robustness.csv',index=False)
    sensitivity=[]
    for overhead in (0.,.001,.005,.01,.02,.05,.1):
        for r in macro.itertuples():sensitivity.append(dict(model=r.model,algorithm=r.algorithm,assumed_selector_loss_charge=overhead,objective=r.objective+(overhead if r.algorithm.startswith('openjev_') else 0)))
    pd.DataFrame(sensitivity).to_csv(out/'selector_cost_sensitivity.csv',index=False)
    t.write_json(out/'verification.json',{'version':'1.4.0','tested_commit':os.getenv('GITHUB_SHA'),'all_gates_passed':True,'model_runs':metas,'matching_model_test_groups':128,'n_algorithms':12,'paired_ledger_rows':len(df),'total_local_selector_forwards':sum(m['new_primary_local_forwards']+m['new_robustness_local_forwards']+m['warmup_forwards'] for m in metas),'rerun_scope':'8 repeat forwards per model; deterministic symbolic replay; not two independent full selector inference runs','run_url':os.getenv('RESEARCH_RUN_URL'),'claim_scope':'Paired pilot conditional on fixed archive, calibration, CPU implementation and selected groups. No universal or proprietary Jev superiority claim.'})
    print(macro.to_string(index=False));print(comp[comp.dataset=='macro_binary'].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('results/openjev'));a=p.parse_args();run(a.inputs,a.out)

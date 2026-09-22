"""Locked v1.6 candidates. All candidate selection precedes test evaluation.

Existing benchmark outcomes are historically inspected: this is exploratory,
not a new external holdout or a proof of real-world policy safety.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import adaptive_router as ar
import transfer_study as t

SEEDS = (1601, 1602, 1603)
METHODS = ('legacy_bellman','compiled_legacy','polarity_bellman','context_bellman',
           'context_polarity','tuning_selected','guarded_selected')


def fraction(group, seed, domain):
    digest = hashlib.sha256(f'dtmoa-opt:{seed}:{domain}:{group}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def split_roles(data, seed):
    roles=[]
    for group, original in zip(data.groups, data.splits):
        if original == 'train':
            role = 'representation' if fraction(group, seed, 'train') < .4 else 'risk_calibration'
        elif original == 'dev':
            role = 'tuning' if fraction(group, seed, 'dev') < .6 else 'promotion_gate'
        else: role = 'test'
        roles.append(role)
    roles=np.array(roles)
    manifest=pd.DataFrame({'dataset': data.dataset, 'sample_id':data.ids.astype(str), 'group':data.groups,
                           'original_split':data.splits,'role':roles,'seed':seed})
    if (manifest.groupby('group').role.nunique() > 1).any(): raise AssertionError('Group leakage')
    parts={role:data.subset(roles==role) for role in set(roles)}
    if any(len(part)<2 for part in parts.values()): raise ValueError('Insufficient split')
    return parts,manifest


def evaluate(policy, data, pool, method, seed, phase):
    is_old=isinstance(policy,t.Policy);is_context=isinstance(policy,ar.ContextRouter)
    began=time.perf_counter()
    bins=policy.bins(data.text) if is_context else None
    head_seconds=time.perf_counter()-began if is_context else 0.
    rows=[]
    for r in range(len(data)):
        responses=data.answers['replacement'][r]
        acquire=lambda i: responses[i]
        if is_old:
            chosen,risk,order,batches,state=policy.execute(acquire)
        else:
            active=policy.leaves[int(bins[r])] if is_context else policy
            chosen,risk,order,batches,state=active.execute(acquire,pool=pool)
        if chosen>=0 and chosen not in order:raise AssertionError('Output without acquisition')
        loss=.25 if chosen<0 else float(data.errors['replacement'][r,chosen])
        rows.append(dict(dataset=data.dataset,sample_id=str(data.ids[r]),group=data.groups[r],phase=phase,seed=seed,method=method,
                         selected=chosen,answered=int(chosen>=0),loss=loss,queries=len(order),cost=.01*len(order),objective=loss+.01*len(order),
                         predicted_error=risk if chosen>=0 else np.nan,query_order=','.join(map(str,order))))
    frame=pd.DataFrame(rows)
    frame.attrs['head_seconds']=head_seconds
    return frame


def group_risk(frame):
    return float(frame.groupby('group').objective.mean().mean())


def fit_candidates(train, parts, dataset, pool, seed):
    candidates={};selection=[];kind='polarity' if dataset in ('strategyqa','jigsaw') else 'equality'
    for strength in (1.,10.,100.):
        old=t.Policy(t.World(train.answers['replacement'],train.errors['replacement'],strength),'bellman',.01)
        candidates[f'legacy:{strength}']=(old,'legacy_bellman',strength)
        signed=ar.compile_joint(ar.fit_joint(train.answers['replacement'],train.errors['replacement'],kind,strength),pool,
                                name=f'polarity:{strength}')
        candidates[f'polarity:{strength}']=(signed,'polarity_bellman',strength)
    for shrinkage in (32.,128.):
        for family,encoding in [('context_bellman','equality'),('context_polarity',kind)]:
            policy=ar.fit_context(parts['representation'],parts['risk_calibration'],'replacement',pool,encoding,shrinkage,32)
            candidates[f'{family}:{shrinkage}']=(policy,family,shrinkage)
    tuning={}
    for name,(policy,family,param) in candidates.items():
        frame=evaluate(policy,parts['tuning'],pool,family,seed,'tuning')
        score=group_risk(frame);tuning[name]=score
        selection.append(dict(dataset=dataset,seed=seed,candidate=name,family=family,parameter=param,tuning_objective=score,
                              context_counts='' if not isinstance(policy,ar.ContextRouter) else str(policy.calibration_counts)))
    chosen={};chosen_names={}
    for family in ('legacy_bellman','polarity_bellman','context_bellman','context_polarity'):
        subset=[(round(tuning[name],12),param,name,policy) for name,(policy,group,param) in candidates.items() if group==family]
        _,_,name,policy=min(subset,key=lambda x:x[:3]);chosen[family]=policy;chosen_names[family]=name
    incumbent=ar.CompiledPolicy.from_legacy(chosen['legacy_bellman'],pool)
    chosen['compiled_legacy']=incumbent;chosen_names['compiled_legacy']=chosen_names['legacy_bellman']
    # Tie preference is incumbent, polarity, context, then combined context/polarity.
    order=('legacy_bellman','polarity_bellman','context_bellman','context_polarity')
    best=min(order,key=lambda f:(round(tuning[chosen_names[f]],12),order.index(f)))
    selected=incumbent if best=='legacy_bellman' else chosen[best]
    chosen['tuning_selected']=selected;chosen_names['tuning_selected']=chosen_names[best]
    new=evaluate(selected,parts['promotion_gate'],pool,'candidate',seed,'promotion_gate')
    old=evaluate(incumbent,parts['promotion_gate'],pool,'incumbent',seed,'promotion_gate')
    newg=new.groupby('group').objective.mean();oldg=old.groupby('group').objective.mean().reindex(newg.index)
    guard=ar.promotion_bound((newg-oldg).to_numpy(),alpha=.05)
    guard.update(dataset=dataset,seed=seed,challenger=chosen_names[best],incumbent=chosen_names['legacy_bellman'],
                 identical_selected_policy=best=='legacy_bellman')
    # Identical incumbent requires no statistical authorization; genuinely new
    # policies require independent gate evidence, not merely low tuning risk.
    chosen['guarded_selected']=selected if guard['promote'] else incumbent
    chosen_names['guarded_selected']=chosen_names[best] if guard['promote'] else chosen_names['legacy_bellman']
    for row in selection:
        row['selected_in_family']=row['candidate']==chosen_names[row['family']]
    gate_ledger=pd.DataFrame({'group':newg.index,'candidate_risk':newg.to_numpy(),'incumbent_risk':oldg.to_numpy()})
    gate_ledger['dataset']=dataset;gate_ledger['seed']=seed
    return chosen,chosen_names,selection,guard,gate_ledger


def save_policy(policy,path,pool):
    if isinstance(policy,t.Policy):policy=ar.CompiledPolicy.from_legacy(policy,pool)
    payload=ar.context_to_json(policy) if isinstance(policy,ar.ContextRouter) else policy.to_json()
    path.write_text(payload)
    restored=(ar.context_from_json(payload,expected_pool=pool) if isinstance(policy,ar.ContextRouter)
              else ar.CompiledPolicy.from_json(payload,expected_pool=pool))
    return restored


def check_compilation(old,compiled,pool):
    for pattern in t.PATTERNS:
        answers=[None if v<0 else str(v) for v in pattern]
        a=old.execute(lambda i:answers[i]);b=compiled.execute(lambda i:answers[i],pool=pool)
        assert a[0]==b[0] and a[2:]==b[2:] and abs(a[1]-b[1])<1e-12
    return len(t.PATTERNS)


def runtime(old,compiled,data,pool,rounds=30):
    samples=data.answers['replacement'][:64]
    def run_one(p):
        for row in samples:
            if p is old:p.execute(lambda i:row[i])
            else:p.execute(lambda i:row[i],pool=pool)
    for _ in range(3):run_one(old);run_one(compiled)
    timings={'legacy':[],'compiled':[]}
    for r in range(rounds):
        for name,p in ([('legacy',old),('compiled',compiled)] if r%2==0 else [('compiled',compiled),('legacy',old)]):
            start=time.perf_counter_ns();run_one(p);elapsed=time.perf_counter_ns()-start
            timings[name].append(elapsed/len(samples))
    return [dict(implementation=name,median_ns=float(np.median(values)),p95_ns=float(np.quantile(values,.95)),
                 n_rounds=rounds,cases_per_round=len(samples),scope='amortized same-process callback execution; excludes fitting and compilation') for name,values in timings.items()]


def summarize(frame):
    rows=[]
    for (ds,seed,method),g in frame.groupby(['dataset','seed','method'],sort=True):
        answered=g[g.answered==1];p=answered.predicted_error.to_numpy();y=answered.loss.to_numpy()
        rows.append(dict(dataset=ds,seed=seed,method=method,n=len(g),groups=g.group.nunique(),objective=g.objective.mean(),
                         group_objective=g.groupby('group').objective.mean().mean(),coverage=g.answered.mean(),
                         accuracy_on_answered=1-answered.loss.mean(),queries=g.queries.mean(),
                         brier=float(np.mean((p-y)**2)) if len(p) else np.nan))
    return pd.DataFrame(rows)


def comparisons(frame, primary_seed=SEEDS[0]):
    contrasts=('polarity_bellman','context_polarity','tuning_selected','guarded_selected');rows=[]
    frame=frame[frame.seed==primary_seed]
    if frame.empty:
        raise ValueError('Primary seed has no test rows')
    for method in contrasts:
        distributions=[];means=[]
        for j,dataset in enumerate(t.PRIMARY_DATASETS):
            x=frame[(frame.dataset==dataset)&(frame.method==method)].set_index('sample_id')
            y=frame[(frame.dataset==dataset)&(frame.method=='legacy_bellman')].set_index('sample_id').reindex(x.index)
            if y.objective.isna().any():raise ValueError('Unpaired cases')
            diff=x.objective-y.objective
            grouped=pd.DataFrame({'group':x.group,'diff':diff}).groupby('group')['diff'].agg(['sum','count'])
            rng=np.random.default_rng(np.random.SeedSequence([primary_seed*100+9,j]));values=[];n=len(grouped)
            for _ in range(20):
                ix=rng.integers(0,n,(100,n));values.extend(grouped['sum'].to_numpy()[ix].sum(1)/grouped['count'].to_numpy()[ix].sum(1))
            lo,hi=np.quantile(values,[.025,.975]);rows.append(dict(dataset=dataset,method=method,difference=diff.mean(),lo=lo,hi=hi,groups=n))
            distributions.append(values);means.append(float(diff.mean()))
        boot=np.mean(distributions,axis=0);lo,hi=np.quantile(boot,[.025,.975]);flo,fhi=np.quantile(boot,[.00625,.99375])
        rows.append(dict(dataset='macro_six',method=method,difference=np.mean(means),lo=lo,hi=hi,family_lo=flo,family_hi=fhi,groups=np.nan))
        # Secondary diagnostic: GQA lacks images; do not drop it from the primary result.
        textboot=np.mean(distributions[:4]+distributions[5:],axis=0);lo,hi=np.quantile(textboot,[.025,.975])
        rows.append(dict(dataset='text_only_sensitivity',method=method,difference=np.mean(means[:4]+means[5:]),lo=lo,hi=hi,groups=np.nan))
    return pd.DataFrame(rows)


def run(cache,out,seeds=SEEDS):
    out.mkdir(parents=True,exist_ok=True);(out/'policies').mkdir(exist_ok=True)
    protocol=json.loads((t.ROOT/'transfer_protocol.json').read_text());new_protocol=t.ROOT/'optimizer_protocol.json'
    frames=[];manifests=[];select_rows=[];guards=[];timing=[];resources=[];exports=[];audit=[];checks=[];gate_ledgers=[]
    for dataset in t.PRIMARY_DATASETS:
        data,source_audit=t.load_dataset(cache/f'{dataset}_predictions.csv',dataset,protocol['source_sha256'][dataset]);audit.append(source_audit)
        pool=ar.Pool(tuple(t.POOL_NAMES['replacement']),protocol['source_revision']+':replacement')
        train=data.subset(data.splits=='train')
        for seed in seeds:
            parts,manifest=split_roles(data,seed);manifests.append(manifest)
            start=time.perf_counter();chosen,names,selected,guard,gate_ledger=fit_candidates(train,parts,dataset,pool,seed)
            fit_seconds=time.perf_counter()-start;gate_ledgers.append(gate_ledger);select_rows.extend(selected);guards.append(guard)
            resources.append(dict(dataset=dataset,seed=seed,fit_seconds=fit_seconds,train_questions=len(train),
                representation_questions=len(parts['representation']),risk_calibration_questions=len(parts['risk_calibration']),
                tuning_questions=len(parts['tuning']),gate_questions=len(parts['promotion_gate']),
                fitting_archive_responses=4*(len(train)+len(parts['tuning'])+len(parts['promotion_gate']))))
            n=check_compilation(chosen['legacy_bellman'],chosen['compiled_legacy'],pool)
            checks.append(dict(dataset=dataset,seed=seed,complete_patterns=n,exact_action_choice_equivalence=True))
            for method in METHODS:
                p=chosen[method];test=evaluate(p,parts['test'],pool,method,seed,'test');frames.append(test)
                path=out/'policies'/f'{dataset}-{seed}-{method}.json';restored=save_policy(p,path,pool)
                replay=evaluate(restored,parts['test'],pool,method,seed,'test')
                if not test[['selected','query_order','objective']].equals(replay[['selected','query_order','objective']]):
                    raise AssertionError('Serialization changed deployed decisions')
                exports.append(dict(dataset=dataset,seed=seed,method=method,selected_candidate=names[method],
                                    bytes=path.stat().st_size,sha256=t.digest(path),roundtrip_exact_decisions=True))
            if seed==seeds[0]:
                timing.extend(dict(dataset=dataset,seed=seed,**r) for r in runtime(chosen['legacy_bellman'],chosen['compiled_legacy'],parts['test'],pool))
            print('completed',dataset,seed,'selected',names['tuning_selected'],'gate',guard['promote'],flush=True)
    frame=pd.concat(frames,ignore_index=True);manifest=pd.concat(manifests,ignore_index=True)
    for seed,g in manifest.groupby('seed'):
        if (g.groupby('group').role.nunique()>1).any():raise AssertionError('Cross-dataset group leakage')
    frame.to_csv(out/'per_case.csv',index=False);manifest.to_csv(out/'split_roles.csv',index=False)
    summary=summarize(frame);summary.to_csv(out/'summary.csv',index=False)
    macro=summary.groupby(['seed','method']).agg(objective=('objective','mean'),coverage=('coverage','mean'),queries=('queries','mean')).reset_index()
    macro.to_csv(out/'macro.csv',index=False);comparisons(frame,seeds[0]).to_csv(out/'comparisons.csv',index=False)
    pd.concat(gate_ledgers,ignore_index=True).to_csv(out/'gate_cases.csv',index=False)
    pd.DataFrame(select_rows).to_csv(out/'selection.csv',index=False);pd.DataFrame(guards).to_csv(out/'promotion_gates.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'runtime.csv',index=False);pd.DataFrame(resources).to_csv(out/'fit_resources.csv',index=False)
    pd.DataFrame(exports).to_csv(out/'exports.csv',index=False);pd.DataFrame(checks).to_csv(out/'compilation_checks.csv',index=False)
    t.write_json(out/'source_audit.json',audit)
    t.write_json(out/'provenance.json',dict(version='1.6.0-candidate',tested_commit=os.getenv('GITHUB_SHA','local'),
        protocol_sha256=t.digest(new_protocol),source_revision=protocol['source_revision'],seeds=list(seeds),
        new_llm_calls=0,independent_new_test_questions=0,old_test_questions=int(frame[(frame.seed==seeds[0])&(frame.method=='legacy_bellman')].shape[0]),
        python=platform.python_version(),run_url=os.getenv('RESEARCH_RUN_URL'),
        limits='Exploratory repeated-use benchmarks; no new neural inference, agent training, visual evidence or individual preferences. Gate is per-dataset conditional, not simultaneous six-task or distribution-shift safety.'))
    print(macro.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=t.ROOT/'.cache/proeval');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--download',action='store_true');parser.add_argument('--main-only',action='store_true')
    args=parser.parse_args()
    if args.download:t.download(args.cache)
    run(args.cache,args.out,(1601,) if args.main_only else SEEDS)

"""Exploratory sensitivity checks, specified after the primary run.
These results must not be labeled preregistered or product measurements.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import study as s

OUT=s.ROOT/'results'

def dependent_optimizer(joint,weight,kind='dp'):
    """V is legal only after a worker has returned a candidate in an earlier batch."""
    return s.optimize(joint,weight,kind=kind,candidate_required=True)

def main():
    truth=s.make_joints(); rows=[]
    for gate in [False,True]:
        policies=s.build_policies(truth,all_policies=False)
        if gate:
            policies['P1_bellman']=[dependent_optimizer(truth[t],s.WEIGHTS[t]) for t in range(5)]
            policies['C3_greedy']=[dependent_optimizer(truth[t],s.WEIGHTS[t],'greedy') for t in range(5)]
        for name,ps in policies.items():
            cs=[s.compile_policy(p) for p in ps]
            if gate:
                for c in cs:
                    for h in c['histories']:
                        prev=0
                        for m in h:
                            if m&(1<<4): assert prev&15
                            prev|=m
            rows.append({'candidate_required':gate,'policy':name,**s.exact_metrics(truth,cs)})
    pd.DataFrame(rows).to_csv(OUT/'exploratory_verifier_dependency.csv',index=False)
    # Updated true-model oracle after drift. This is NOT a retrained deployed model.
    drift=[]
    for scenario in s.CFG['stress_tests']:
        joints=s.make_joints(scenario); cost=s.COST.copy(); lat=s.LAT.copy(); probs=s.PROBS
        if scenario=='cost_shift': cost[4]*=8; cost[5]*=4; lat[5]*=5
        if scenario=='easy_only_shift':
            joints[0]=s.factual_joint(perfect_easy=True); probs=np.array([1.,0,0,0,0])
        ps=[s.optimize(joints[t],s.WEIGHTS[t],cost=cost,latency=lat) for t in range(5)]
        cs=[s.compile_policy(p,cost,lat) for p in ps]
        drift.append({'scenario':scenario,'policy':'updated_oracle',**s.exact_metrics(joints,cs,probs)})
    pd.DataFrame(drift).to_csv(OUT/'exploratory_updated_oracle.csv',index=False)
    # Full exact main-policy action utilization, averaged over fitted replicates.
    traces=[]
    for rep in range(s.CFG['replicates']):
        with np.load(OUT/'raw'/f'replicate_{rep:02d}.npz') as f: fitted=f['fitted_joint']
        policies=s.build_policies(fitted,all_policies=False)
        for name,ps in policies.items():
            for t,p in enumerate(ps):
                c=s.compile_policy(p); obs_prob=truth[t].sum(axis=0)
                row={'replicate':rep,'policy':name,'type':s.TYPE_NAMES[t]}
                for a,label in enumerate(s.CFG['channels']):
                    row[label]=float(np.dot(obs_prob,(c['mask']&(1<<a))>0))
                row['no_acquisition']=float(np.dot(obs_prob,c['calls']==0))
                traces.append(row)
    pd.DataFrame(traces).to_csv(OUT/'action_utilization.csv',index=False)
    # Accuracy/Brier supplements, no replacement of the primary objective.
    summary={
      'status':'Exploratory: specified after reading main results',
      'reason':'Primary V directly outputs a grounded binary answer. Check sensitivity to candidate-before-verification feasibility and frozen-vs-updated likelihoods.',
      'independence_ablation_note':'Fitted independence approximation was slightly better on mean J; do not claim joint correlation modeling is necessary in this benchmark.',
      'novelty':'Bellman recursion is standard; no new general optimality theorem is claimed.'}
    (OUT/'exploratory_status.json').write_text(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows)[['candidate_required','policy','objective','cost','latency']].to_string(index=False))
    print(pd.DataFrame(drift)[['scenario','objective']].to_string(index=False))

if __name__=='__main__': main()

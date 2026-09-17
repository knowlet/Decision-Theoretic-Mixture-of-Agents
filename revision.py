"""v1.1 exploratory analyses. All results are computed, not prefilled.

Usage: OPENBLAS_NUM_THREADS=1 python revision.py --results results
Matched analyses reuse v1.0 calibration and held-out draws; learning curves use
fresh independent calibration streams and exact expected-risk evaluation.
"""
from __future__ import annotations
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import study as s

CFG = json.loads((s.ROOT / 'revision_protocol.json').read_text())

def static_all(joint, weight):
    """Best ONE batch over ALL six sources, including stop. No later adaptation."""
    mass, post, decisions, risk = s.terminal(joint, weight)
    _, _, charges = s.batch_terms()
    best, chosen = float(risk[0]), 0
    for mask, children in s.TRANS[0]:
        value = charges[mask] + np.dot(mass[children], risk[children])
        if value < best - 1e-12:
            best, chosen = float(value), mask
    actions = np.zeros(3**s.N, dtype=np.int16)
    actions[0] = chosen
    return s.Policy(actions, decisions, best, post)

def extra_policies(joints):
    out = {name: [] for name in CFG['matched_controls']}
    for j, w in zip(joints, s.WEIGHTS):
        out['R0_stop'].append(s.optimize(j,w,budget=0))
        out['R1_static_all'].append(static_all(j,w))
        out['R2_greedy_workers'].append(s.optimize(j,w,'greedy',allowed=15))
        out['R3_dp_workers'].append(s.optimize(j,w,allowed=15))
    return out

def per_case_metrics(compiled, ts, ys, os):
    out = {}
    for t, c in enumerate(compiled):
        idx = ts == t
        for name, v in s.result_vectors(c,s.WEIGHTS[t]).items():
            out[name] = out.get(name, 0.) + v[ys[idx],os[idx]].sum()/len(ts)
    out['selective_accuracy'] = 1-out['error']/out['coverage'] if out['coverage'] else np.nan
    return out

def summary(frame):
    rows=[]
    for name, group in frame.groupby('policy',sort=False):
        row={'policy':name}
        for col in ('objective','terminal_loss','cost','latency','coverage'):
            m,lo,hi=s.replicate_t_ci(group[col]); row.update({col:m,col+'_lo':lo,col+'_hi':hi})
        rows.append(row)
    return pd.DataFrame(rows)

def run(results):
    out=results/'revision';out.mkdir(exist_ok=True)
    truth=s.make_joints(); rows=[]; exact=[]; route_rows=[]; coupling_rows=[]
    for rep in range(CFG['matched_replicates']):
        with np.load(results/'raw'/f'replicate_{rep:02d}.npz') as f:
            fitted=f['fitted_joint'];ts=f['type'];ys=f['y'];os=f['outcome']
        ps=extra_policies(fitted)
        for name, policies in ps.items():
            cs=[s.compile_policy(p) for p in policies]
            rows.append({'replicate':rep,'policy':name,**per_case_metrics(cs,ts,ys,os)})
            exact.append({'replicate':rep,'policy':name,**s.exact_metrics(truth,cs)})
        main=s.build_policies(fitted,all_policies=False)
        main['R1_static_all']=ps['R1_static_all']
        for name, policies in main.items():
            cs=[s.compile_policy(p) for p in policies]
            # Route noise selects a wrong type uniformly among the four others.
            # Actual loss/label remain hidden from the controller; evaluator uses them.
            risk=np.array([[np.sum(truth[t]*s.result_vectors(cs[r],s.WEIGHTS[t])['objective'])
                            for r in range(5)] for t in range(5)])
            for eta in CFG['route_error_rates']:
                conf=np.full((5,5),eta/4);np.fill_diagonal(conf,1-eta)
                value=float(np.sum(s.PROBS[:,None]*conf*risk))
                route_rows.append({'replicate':rep,'policy':name,'route_error':eta,'objective':value})
        for coupling in CFG['verifier_coupling']:
            target=truth.copy()
            for t in range(3):target[t]=s.factual_joint(easy=t==0,verifier_coupling=coupling)
            for name in ('C3_greedy','P0_heuristic','P1_bellman','R1_static_all'):
                coupling_rows.append({'replicate':rep,'coupling':coupling,'policy':name,
                    **s.exact_metrics(target,[s.compile_policy(p) for p in main[name]])})
    df=pd.DataFrame(rows);df.to_csv(out/'matched_by_replicate.csv',index=False)
    summary(df).to_csv(out/'matched_summary.csv',index=False)
    pd.DataFrame(exact).to_csv(out/'matched_exact.csv',index=False)
    pd.DataFrame(route_rows).to_csv(out/'routing_noise.csv',index=False)
    pd.DataFrame(coupling_rows).to_csv(out/'verifier_coupling.csv',index=False)
    primary=pd.read_csv(results/'metrics_by_replicate.csv')
    both=pd.concat([primary,df],ignore_index=True)
    pivot=both.pivot(index='replicate',columns='policy',values='objective')
    comparisons=[]
    for a,b in [('P1_bellman','R1_static_all'),('R3_dp_workers','R2_greedy_workers'),
                ('P1_bellman','R3_dp_workers'),('C3_greedy','R2_greedy_workers')]:
        difference=pivot[a]-pivot[b];m,lo,hi=s.replicate_t_ci(difference)
        comparisons.append({'a':a,'b':b,'difference':m,'lo':lo,'hi':hi,
                            'p_unadjusted':float(stats.ttest_1samp(difference,0).pvalue)})
    # Explicitly exploratory; adjusted p-values do not make this preregistration.
    last=0
    for rank,i in enumerate(sorted(range(len(comparisons)),key=lambda i:comparisons[i]['p_unadjusted'])):
        last=max(last,min(1.,(len(comparisons)-rank)*comparisons[i]['p_unadjusted']))
        comparisons[i]['holm_p_exploratory']=last
    pd.DataFrame(comparisons).to_csv(out/'matched_comparisons.csv',index=False)
    learning=[]
    for n in CFG['calibration_sizes_per_type']:
        for rep in range(CFG['learning_replicates']):
            rng=np.random.default_rng(np.random.SeedSequence([CFG['learning_seed'],n,rep]))
            fitted=np.array([s.calibrate(j,n,rng) for j in truth])
            for kind,name in [('greedy','C3_greedy'),('dp','P1_bellman'),('independent','P1_independence')]:
                policies=[s.optimize(s.independent_approx(fitted[t]) if kind=='independent' else fitted[t],
                                     s.WEIGHTS[t],kind='greedy' if kind=='greedy' else 'dp') for t in range(5)]
                learning.append({'n_per_type':n,'replicate':rep,'policy':name,
                                 **s.exact_metrics(truth,[s.compile_policy(p) for p in policies])})
        print(f'learning curve n={n} done',flush=True)
    pd.DataFrame(learning).to_csv(out/'calibration_curve.csv',index=False)
    # Standalone CPU measurements, NOT normalized latency or end-to-end LLM time.
    timings=[]
    for rep in range(CFG['runtime_repeats']):
        for kind,name in [('greedy','C3_greedy'),('greedy_batch','C3_greedy_batch'),('dp','P1_bellman')]:
            start=time.perf_counter()
            for t in range(5):s.optimize(truth[t],s.WEIGHTS[t],kind)
            timings.append({'repeat':rep,'policy':name,'five_type_plan_seconds':time.perf_counter()-start})
    pd.DataFrame(timings).to_csv(out/'planning_runtime.csv',index=False)
    # Exact full-support preference information bound with partial informativeness.
    info=[]
    for accuracy in [.5,.6,.75,.9,.99,1.]:
        j=s.preference_joint(accuracy)
        marginal=np.zeros((2,2))
        for bit in (0,1):marginal[:,bit]=j[:,s.BITS[:,5]==bit].sum(axis=1)
        channel_tv=.5*np.abs(marginal[0]/.5-marginal[1]/.5).sum()
        bayes_success=marginal.max(axis=0).sum()
        assert abs(bayes_success-(1+channel_tv)/2)<1e-12
        info.append({'elicitor_accuracy':accuracy,'channel_total_variation':channel_tv,
                     'best_always_answer_accuracy':bayes_success})
    pd.DataFrame(info).to_csv(out/'preference_information_bound.csv',index=False)
    (out/'status.json').write_text(json.dumps({**CFG,'matching':'reused v1.0 held-out cases; no threshold tuning',
       'learning_evaluation':'fresh calibration draws and exact expectations, not new LLM calls',
       'plan_overhead_break_even':float((pivot.C3_greedy-pivot.P1_bellman).mean()),
       'warning':'Sampling intervals exclude uncertainty about the realism of the generator.'},indent=2))
    print(summary(df)[['policy','objective','cost','latency']].to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=s.ROOT/'results')
    run(ap.parse_args().results)

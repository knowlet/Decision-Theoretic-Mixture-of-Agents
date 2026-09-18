"""Post-release diagnostic: action-ID flips are not necessarily answer flips.
No new model calls, parameter fitting changes, or primary result rewrites.
Requires published inputs/*/order_robustness.csv and the pinned ProEval files.
"""
import argparse
from pathlib import Path
import pandas as pd
import openjev_benchmark as b

def output_and_loss(action,case):
    if action=='defer':return None,.25
    if not action.startswith('emit_'):raise ValueError('Unknown action')
    index=int(action.split('_')[1])
    if index not in case['panel']:raise ValueError('Unacquired answer')
    return case['display_answers'][str(index)],float(case['evaluation_errors'][index])

def run(inputs,cache,out):
    cases,*_=b.prepare(cache);mapping={c['id']:c for c in cases};rows=[]
    for model in b.MODELS:
        frame=pd.read_csv(inputs/model/'order_robustness.csv')
        if len(frame)!=32 or frame.id.duplicated().any():raise ValueError('Probe set changed')
        for r in frame.itertuples():
            case=mapping[r.id]
            if case['split']!='test':raise ValueError('Non-test probe')
            a,la=output_and_loss(r.original,case);z,lz=output_and_loss(r.reversed,case)
            rows.append(dict(model=model,id=r.id,dataset=r.dataset,action_changed=r.original!=r.reversed,
                answer_changed=a!=z,deferral_changed=(a is None)!=(z is None),original_loss=la,reversed_loss=lz,loss_changed=la!=lz))
    detail=pd.DataFrame(rows);summary=detail.groupby('model').agg(n=('id','size'),action_flips=('action_changed','sum'),
        answer_or_defer_flips=('answer_changed','sum'),defer_flips=('deferral_changed','sum'),loss_flips=('loss_changed','sum'),
        original_loss=('original_loss','mean'),reversed_loss=('reversed_loss','mean')).reset_index()
    out.mkdir(parents=True,exist_ok=True);detail.to_csv(out/'semantic_order_cases.csv',index=False);summary.to_csv(out/'semantic_order_summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,default=Path('inputs'));p.add_argument('--cache',type=Path,default=Path('.cache/proeval'));p.add_argument('--out',type=Path,default=Path('results/order-audit'));a=p.parse_args();run(a.inputs,a.cache,a.out)

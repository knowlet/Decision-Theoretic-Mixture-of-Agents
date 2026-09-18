"""Post-primary direction/class diagnostic; does not tune or change policies."""
import argparse,json
from pathlib import Path
import pandas as pd
import transfer_study as t


def run(cache,out):
    protocol=json.loads((t.ROOT/'transfer_protocol.json').read_text())
    scores=pd.read_csv(out/'per_case.csv',dtype={'sample_id':str})
    rows=[];training=[]
    for dataset in ('strategyqa','jigsaw'):
        path=cache/f'{dataset}_predictions.csv'
        data,_=t.load_dataset(path,dataset,protocol['source_sha256'][dataset])
        raw=pd.read_csv(path,dtype=str,keep_default_na=False)
        truth={r['index']:t.normalize_gold(r['ground_truth'],dataset) for r in raw.to_dict('records')}
        lookup={str(key):i for i,key in enumerate(data.ids)}
        for i,name in enumerate(t.POOL_NAMES['replacement']):
            for answer in ('yes','no'):
                ix=(data.splits=='train')&(data.answers['replacement'][:,i]==answer)
                training.append(dict(dataset=dataset,worker=name,answer=answer,n=int(ix.sum()),
                    archived_error=float(data.errors['replacement'][ix,i].mean()) if ix.any() else None))
        for (seed,method),group in scores[scores.dataset==dataset].groupby(['seed','method']):
            for gold in ('yes','no'):
                n=answered=correct=wrong=deferred=0
                for r in group.itertuples():
                    if truth[r.sample_id]!=gold:continue
                    n+=1
                    if r.selected<0:deferred+=1;continue
                    answer=data.answers['replacement'][lookup[r.sample_id],r.selected]
                    answered+=1;correct+=int(answer==gold);wrong+=int(answer!=gold)
                rows.append(dict(dataset=dataset,seed=seed,method=method,gold=gold,n=n,answered=answered,
                    correct=correct,wrong=wrong,deferred=deferred,coverage=answered/n if n else None,
                    selected_error=wrong/answered if answered else None))
    pd.DataFrame(training).to_csv(out/'direction_training_audit.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'class_outcomes.csv',index=False)
    t.write_json(out/'diagnostic_scope.json',{'status':'Post-primary descriptive audit; no hyperparameter, objective or policy changes',
        'ground_truth':'Same pinned archive gold, not an independent human audit',
        'jigsaw':'yes means toxic under the existing >0.5 annotator-fraction rule; no individual annotator preference inference'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=t.ROOT/'.cache/proeval');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.out)

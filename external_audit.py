"""Post-primary source-quality audit and frozen zero-to-five-shot shift.

Gold is accessed here only after primary policies have been selected and scored.
Original primary results are never rewritten. No language-model endpoints.
"""
from __future__ import annotations
import argparse,hashlib,json,re,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import external_replay as e

GOLD_REV='c30699e8356da336a370243923dbaf21066bb9fe'
GOLD_SHA='74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7'
FIVE_SHA='fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe'

def verified_download(url,path,digest):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix('.partial');urllib.request.urlretrieve(url,tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=digest:
            tmp.unlink(missing_ok=True);raise ValueError('Hash mismatch')
        tmp.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Existing source hash mismatch')
    return path

def normalized(s):return ' '.join(str(s).strip().casefold().split())

def prompt_key(prompt,subject):
    text=e.unwrap(prompt).strip()
    prefix='Please answer with the letter of the correct answer.'
    if text.startswith(prefix):text=text[len(prefix):].strip()
    text=text.split('\nPrint only a single choice')[0].strip()
    parts=re.split(r'\n([A-D])\)\s*',text)
    if len(parts)!=9 or parts[1::2]!=list('ABCD'):return None
    return (subject.replace('mmlu-','').replace('-','_'),normalized(parts[0]),tuple(normalized(s) for s in parts[2::2]))

def gold_index(frame):
    index={}
    for r in frame.itertuples():
        key=(str(r.subject),normalized(r.question),tuple(normalized(x) for x in r.choices))
        index.setdefault(key,set()).add(int(r.answer))
    return {k:next(iter(v)) for k,v in index.items() if len(v)==1},sum(len(v)>1 for v in index.values())

def permuted_gold_index(frame):
    # Map by option text, not raw letters, when answer choices are permuted.
    index={}
    for r in frame.itertuples():
        choices=tuple(normalized(x) for x in r.choices)
        if len(set(choices))!=4:continue
        key=(str(r.subject),normalized(r.question),tuple(sorted(choices)))
        index.setdefault(key,set()).add(choices[int(r.answer)])
    return {k:next(iter(v)) for k,v in index.items() if len(v)==1},sum(len(v)>1 for v in index.values())

def match_permuted_gold(key,lookup):
    if key is None:return None
    subject,question,choices=key
    if len(choices)!=4 or len(set(choices))!=4:return None
    correct=lookup.get((subject,question,tuple(sorted(choices))))
    return choices.index(correct) if correct in choices else None

def audit_gold(raw,results,gold_path):
    gold=pd.read_parquet(gold_path);lookup,ambiguous=permuted_gold_index(gold)
    exact_lookup,_=gold_index(gold);exact_matches=0;reordered_matches=0
    rows=[];missing=[]
    for _,r in raw[raw.eval_name.str.startswith('mmlu-')].iterrows():
        key=prompt_key(r.prompt,str(r.eval_name))
        answer=match_permuted_gold(key,lookup)
        if answer is None:missing.append(str(r.sample_id));continue
        if key in exact_lookup:exact_matches+=1
        else:reordered_matches+=1
        for i,m in enumerate(e.MODELS):
            parsed=e.parse_answer(r[m+'|model_response']);old=float(r[m]);new=float(parsed==answer)
            rows.append((str(r.sample_id),m,i,parsed,answer,old,new,int(parsed!=4),int(old!=new)))
    frame=pd.DataFrame(rows,columns=['sample_id','model','model_index','parsed_answer','gold_answer','upstream_score','strict_gold_score','parseable','score_difference'])
    frame.to_csv(results/'mmlu_gold_pairs.csv',index=False)
    matched=frame.sample_id.nunique();eligible=int(raw.eval_name.str.startswith('mmlu-').sum())
    if matched<.95*eligible:raise AssertionError(f'Insufficient independent gold matching: {matched}/{eligible}')
    # Re-score the already-selected actions, without retraining or changing primary data.
    primary=pd.read_csv(results/'primary_per_case.csv')
    prim=primary[(primary.family=='mmlu')&primary.sample_id.isin(set(frame.sample_id))].copy()
    scores=frame.pivot(index='sample_id',columns='model_index',values='strict_gold_score')
    prim['original_selected_score']=prim.selected_score
    prim['selected_score']=[np.nan if i<0 else scores.loc[s,int(i)] for s,i in zip(prim.sample_id,prim.selected_model_index)]
    prim['terminal_loss']=np.where(prim.answered==1,1-prim.selected_score,prim.defer_loss)
    prim['objective']=prim.terminal_loss+prim.lambda_usd*prim.historical_cost_usd
    prim['scenario']='mmlu_gold_regrade_frozen'
    prim.to_csv(results/'mmlu_gold_regraded_per_case.csv',index=False)
    e.summary(prim).to_csv(results/'mmlu_gold_regraded_summary.csv',index=False)
    report={'status':'Post-primary exploratory independent-reference audit; no primary results modified',
        'source':'cais/mmlu','source_revision':GOLD_REV,'source_sha256':GOLD_SHA,
        'original_mmlu_rows':eligible,'matched_rows':int(matched),'unmatched_rows':len(missing),'ambiguous_reference_keys':ambiguous,
        'exact_order_matches':exact_matches,'reordered_choice_matches':reordered_matches,
        'evaluated_model_response_pairs':len(frame),'parseable_pairs':int(frame.parseable.sum()),
        'parseable_score_disagreements':int(frame.loc[frame.parseable==1,'score_difference'].sum()),
        'all_strict_score_disagreements':int(frame.score_difference.sum()),
        'unmatched_sample_ids':missing,
        'grading_rule':'Strict parsed leading answer compared with independently matched subject+question+choice set; correct option TEXT is mapped into the trace order, never by reusing a reference letter after permutation. Unparsed responses count incorrect only in this secondary strict regrade, not retroactively in primary scores.',
        'limitations':'MMLU only. Matching original benchmark labels does not exclude wrong/ambiguous benchmark gold or pretraining contamination.'}
    (results/'mmlu_gold_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

def freeze_primary(data,results):
    tr=data.subset(data.splits=='train')
    selected=pd.read_csv(results/'development_selection.csv')
    selected=selected[(selected.scenario=='in_distribution')&(selected.lambda_usd==10.)&(selected.defer_loss==.25)]
    policies={}
    for r in selected.itertuples():
        f=e.fit(tr.answers,tr.scores,tr.costs,float(r.prior_strength))
        policies[r.policy]=e.build_policy(f,r.policy,10.,.25)
    assert set(policies)==set(e.POLICIES)
    return policies

def five_shot(zero,five_raw,results):
    d5,audit=e.prepare(five_raw)
    original=zero.subset(zero.splits=='test')
    map5={sid:i for i,sid in enumerate(d5.ids)}
    ids=[sid for sid in original.ids if sid in map5]
    ix=[map5[sid] for sid in ids]
    target=d5.subset(np.array(ix))
    original_map={s:i for i,s in enumerate(original.ids)}
    iz=np.array([original_map[s] for s in ids])
    target.groups=original.groups[iz];target.splits=np.full(len(target),'test')
    if not np.array_equal(target.family,original.family[iz]):raise AssertionError('Task identities changed')
    if set(target.groups)&set(zero.groups[zero.splits!='test']):raise AssertionError('Shift train/test overlap')
    policies=freeze_primary(zero,results)
    frame=pd.concat([e.evaluate(p,target,10.,.25,'zero_to_five_shot_frozen') for p in policies.values()],ignore_index=True)
    frame.to_csv(results/'five_shot_per_case.csv',index=False)
    e.summary(frame).to_csv(results/'five_shot_summary.csv',index=False)
    e.paired_bootstrap(frame).to_csv(results/'five_shot_comparisons.csv',index=False)
    report={'source_revision':e.REVISION,'source_sha256':FIVE_SHA,'zero_shot_test_rows':len(original),
       'matched_test_rows':len(target),'unmatched_test_rows':len(original)-len(target),
       'target_model_responses':len(target)*len(e.MODELS),'parameters_retrained_on_five_shot':False,
       'zero_shot_non_test_group_overlap':0,'five_shot_loading_audit':audit,
       'limitations':'Sample IDs anchor target question identity. Few-shot demonstration overlap with calibration questions is not audited; no claim of zero demonstration overlap. Latency not present. Historical estimated cost increases with prompt length.'}
    (results/'five_shot_provenance.json').write_text(json.dumps(report,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x))+'\n')
    print(e.summary(frame).to_string(index=False),flush=True)

def main(source,results,cache):
    raw=e.load_source(source);zero,_=e.prepare(raw)
    gold_url=f'https://huggingface.co/datasets/cais/mmlu/resolve/{GOLD_REV}/all/test-00000-of-00001.parquet?download=true'
    gold=verified_download(gold_url,cache/'mmlu_test.parquet',GOLD_SHA)
    audit_gold(raw,results,gold)
    five_url=f'https://huggingface.co/datasets/withmartian/routerbench/resolve/{e.REVISION}/routerbench_5shot.pkl?download=true'
    five=verified_download(five_url,cache/'routerbench_5shot.pkl',FIVE_SHA)
    with five.open('rb') as f:d5=e.RestrictedUnpickler(f).load()
    five_shot(zero,d5,results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=e.ROOT/'.cache/routerbench_0shot.pkl')
    p.add_argument('--results',type=Path,default=e.ROOT/'results/external');p.add_argument('--cache',type=Path,default=e.ROOT/'.cache')
    a=p.parse_args();main(a.source,a.results,a.cache)

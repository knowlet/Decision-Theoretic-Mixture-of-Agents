"""Runtime-only sharding of the locked pilot with explicit CPU accumulation.
Primary cases/options/algorithms/labels stay unchanged. Warmup is per worker.
No numerical-identity claim is made against the native BF16 CPU implementation.
"""
from __future__ import annotations
import argparse,json,os,platform,resource,sys
from pathlib import Path
import numpy as np
import pandas as pd
import openjev_benchmark as b
import transfer_study as t
# Shard counts scale with the locked benchmark size so each runner keeps the
# v1.4 per-shard forward budget; v1.5 locks six datasets instead of four.
SHARDS={'qwen35-4b':12,'qwen35-0.8b':3}

def backend():
    import torch
    import torch.nn.functional as F
    old_linear=torch.nn.Linear.forward;old_conv=torch.nn.Conv1d.forward
    def linear(self,x):
        if x.device.type=='cpu' and x.dtype==torch.bfloat16:
            return F.linear(x.float(),self.weight.float(),None if self.bias is None else self.bias.float()).to(x.dtype)
        return old_linear(self,x)
    def conv(self,x):
        if x.device.type=='cpu' and x.dtype==torch.bfloat16:
            return self._conv_forward(x.float(),self.weight.float(),None if self.bias is None else self.bias.float()).to(x.dtype)
        return old_conv(self,x)
    torch.nn.Linear.forward=linear;torch.nn.Conv1d.forward=conv

def cpu_name():
    return next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')

def shard(label,index,cache,out,vendor):
    count=SHARDS[label]
    if index not in range(count):raise ValueError('Unknown shard')
    out.mkdir(parents=True,exist_ok=True);vendor=vendor.resolve();sys.path.insert(0,str(vendor/'src'))
    from openjev_phase1.direct import score,encode_prompt
    import openjev_phase1.direct as module
    import hashlib
    path=Path(module.__file__).resolve();content=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    assert vendor in path.parents and blob=='943f34728d1966bf2325d6e6d34d2a7043cd6a56'
    cases,baselines,selection,resources=b.prepare(cache)
    assigned=[c for i,c in enumerate(cases) if i%count==index]
    reverse_ids={c['id'] for ds in b.DATASETS for c in [x for x in cases if x['dataset']==ds and x['split']=='test'][:b.ORDER_N]}
    repeat_ids={c['id'] for ds in b.DATASETS for c in [x for x in cases if x['dataset']==ds and x['split']=='test'][:b.REPEAT_N]}
    backend();model,tokenizer,metadata=b.cpu_model(label)
    metadata.update(cpu_backend='BF16 stored weights and outputs; Linear/Conv FP32 accumulation',cpu_backend_sha256=t.digest(Path(__file__)),shard=index,shards=count,cpu_model=cpu_name())
    requests=[b.render_request(c) for c in assigned]
    lengths=[len(encode_prompt(tokenizer,r,4096)[0]) for r in requests]
    score(model,tokenizer,requests[0],metadata,4096)
    records=[]
    with (out/'scores.jsonl').open('w') as dst:
        for i,c in enumerate(assigned):
            for kind in ['primary']+(['reverse'] if c['id'] in reverse_ids else [])+(['repeat'] if c['id'] in repeat_ids else []):
                request=b.render_request(c,kind=='reverse');r=score(model,tokenizer,request,metadata,4096);b.validate_result(r,request)
                record={'case_id':c['id'],'kind':kind,'request_sha256':b.hash_object(request),'result':r}
                records.append(record);dst.write(json.dumps(record,allow_nan=False)+'\n');dst.flush()
            print(label,index,'completed',i+1,'/',len(assigned),flush=True)
    t.write_json(out/'provenance.json',{'model':label,'shard':index,'shards':count,'tested_commit':os.getenv('GITHUB_SHA'),'metadata':metadata,
        'upstream_direct_git_blob':blob,'request_spec_sha256':t.digest(b.ROOT/'openjev_protocol.json'),'primary':len(assigned),'forwards':len(records),'warmup':1,
        'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'input_tokens_min':min(lengths),'input_tokens_max':max(lengths),'run_url':os.getenv('RESEARCH_RUN_URL')})
    t.write_json(out/'SHA256.json',{p.name:t.digest(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256.json'})

def finalize(label,incoming,cache,out):
    out.mkdir(parents=True,exist_ok=True);count=SHARDS[label];records=[];provenance=[]
    for index in range(count):
        candidates=list(incoming.glob(f'openjev-shard-{label}-{index}-*/results/shard'))
        if len(candidates)!=1:raise ValueError(f'Missing or duplicated shard {label}:{index}: {candidates}')
        path=candidates[0];checks=json.loads((path/'SHA256.json').read_text())
        for name,sha in checks.items():
            if Path(name).name!=name or t.digest(path/name)!=sha:raise ValueError('Bad shard digest')
        p=json.loads((path/'provenance.json').read_text());assert p['shard']==index and p['model']==label and p['shards']==count
        assert p['tested_commit']==os.getenv('GITHUB_SHA') and p['request_spec_sha256']==t.digest(b.ROOT/'openjev_protocol.json')
        provenance.append(p);records.extend(json.loads(x) for x in (path/'scores.jsonl').read_text().splitlines())
    cases,baselines,selection,resources=b.prepare(cache);case_map={c['id']:c for c in cases}
    buckets={k:{} for k in ('primary','reverse','repeat')}
    for record in records:
        case_id,kind=record['case_id'],record['kind']
        if kind not in buckets or case_id in buckets[kind] or case_id not in case_map:raise ValueError('Unexpected/duplicate inference')
        request=b.render_request(case_map[case_id],kind=='reverse')
        if record['request_sha256']!=b.hash_object(request):
            raise AssertionError('Request hash mismatch for '+case_id+' ('+kind+'); re-rendered candidates: '+repr(request['state']['candidates']))
        b.validate_result(record['result'],request);buckets[kind][case_id]=record['result']
    outputs=buckets['primary'];assert set(outputs)==set(case_map) and len(buckets['reverse'])==b.ORDER_N*len(b.DATASETS) and len(buckets['repeat'])==b.REPEAT_N*len(b.DATASETS)
    baselines.to_csv(out/'baselines.csv',index=False);t.write_json(out/'selection.json',selection);t.write_json(out/'calibration_resources.json',resources)
    t.write_json(out/'case_manifest.json',[{k:c[k] for k in ('id','dataset','sample_id','group','split','panel')}|{'request_sha256':b.hash_object(b.render_request(c))} for c in cases])
    with (out/'selector_scores.jsonl').open('w') as dst:
        for c in cases:dst.write(json.dumps(outputs[c['id']],allow_nan=False)+'\n')
    dev=[c for c in cases if c['split']=='dev'];features=[];correct=[]
    for c in dev:
        candidate,x,_=b.candidate_proposal(outputs[c['id']]);features.append(x);correct.append(1-c['evaluation_errors'][candidate])
    calibrator=b.SuccessCalibrator().fit(features,correct);t.write_json(out/'success_calibrator.json',calibrator.metadata()|{'n_development':len(dev),'development_ids':[c['id'] for c in dev]})
    rows=[];confidence=[]
    for c in cases:
        if c['split']!='test':continue
        r=outputs[c['id']];candidate,x,rawq=b.candidate_proposal(r);pcal=float(calibrator.predict([x])[0]);action=b.validate_result(r,b.render_request(c));raw=-1 if action=='defer' else int(action.split('_')[1])
        confidence.append(dict(id=c['id'],dataset=c['dataset'],group=c['group'],candidate=candidate,correct=1-c['evaluation_errors'][candidate],raw_option_score=rawq,calibrated_success=pcal))
        for algorithm,selected in [('openjev_raw_fixed3',raw),('openjev_calibrated_fixed3',candidate if pcal>=1-b.DEFER else -1)]:
            assert selected in [-1,*c['panel']]
            err=b.DEFER if selected<0 else float(c['evaluation_errors'][selected])
            rows.append(dict(dataset=c['dataset'],sample_id=c['sample_id'],group=c['group'],algorithm=algorithm,selected=selected,answered=int(selected>=0),loss=err,queries=3,cost=.03,objective=err+.03,controller_seconds=r['total_seconds'],new_selector_calls=1,input_tokens=r['input_tokens']))
    df=pd.concat([baselines,pd.DataFrame(rows)],ignore_index=True);df['selector_model']=label;df.to_csv(out/'per_case.csv',index=False);pd.DataFrame(confidence).to_csv(out/'confidence.csv',index=False)
    reverse=[];repeat=[]
    for kind in ('reverse','repeat'):
        for cid,r in buckets[kind].items():
            c=case_map[cid];old=outputs[cid];assert old['model']['shard']==r['model']['shard']
            a=b.validate_result(old,b.render_request(c));bb=b.validate_result(r,b.render_request(c,kind=='reverse'))
            p=dict(zip(old['option_ids'],old['probabilities']));q=dict(zip(r['option_ids'],r['probabilities']))
            if kind=='reverse':reverse.append(dict(id=cid,dataset=c['dataset'],original=a,reversed=bb,flipped=int(a!=bb),tv=.5*sum(abs(p[k]-q[k]) for k in p),forward_seconds=r['total_seconds']))
            else:repeat.append(dict(id=cid,max_probability_delta=max(abs(p[k]-q[k]) for k in p),same_argmax=a==bb))
    assert all(r['same_argmax'] and r['max_probability_delta']<1e-5 for r in repeat)
    pd.DataFrame(reverse).to_csv(out/'order_robustness.csv',index=False);t.write_json(out/'repeat_check.json',repeat)
    t.write_json(out/'provenance.json',{'version':'1.5.0','tested_commit':os.getenv('GITHUB_SHA'),'selector_model':label,'metadata':provenance[0]['metadata'],'all_worker_provenance':provenance,
        'request_spec_sha256':t.digest(b.ROOT/'openjev_protocol.json'),'upstream_direct_git_blob':provenance[0]['upstream_direct_git_blob'],
        'n_datasets':len(b.DATASETS),'n_test':b.TEST_N*len(b.DATASETS),'n_development':b.DEV_N*len(b.DATASETS),
        'new_primary_local_forwards':len(cases),'new_robustness_local_forwards':(b.ORDER_N+b.REPEAT_N)*len(b.DATASETS),'warmup_forwards':count,
        'new_llm_api_calls':0,'jev_proprietary_calls':0,'cera_training_steps':0,'worker_generation_calls':0,'run_url':os.getenv('RESEARCH_RUN_URL'),
        'scope':'CPU-accumulation compatibility variant of unchanged upstream direct scorer. Same locked primary cases; warmups counted per worker. Archived workers, new local selector inference only.'})
    t.write_json(out/'SHA256.json',{p.name:t.digest(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256.json'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['shard','finalize']);p.add_argument('--model',choices=b.MODELS,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--cache',type=Path,default=t.ROOT/'.cache/proeval');p.add_argument('--out',type=Path,required=True);p.add_argument('--vendor',type=Path,default=t.ROOT/'vendor/openjev');p.add_argument('--incoming',type=Path,default=Path('incoming'));p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download:t.download(a.cache)
    if a.mode=='shard':shard(a.model,a.index,a.cache,a.out,a.vendor)
    else:finalize(a.model,a.incoming,a.cache,a.out)

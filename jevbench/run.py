"""Real checkpoint inference; workers must not receive the separate gold artifact."""
from __future__ import annotations
import argparse,json,os,platform,resource,time
from pathlib import Path
import numpy as np
from .adapters import Adapter,MODELS,sha
from .fixtures import read_jsonl,digest,write

def validate_score(score,request):
    ids=[o['id'] for o in request['options']];p=np.asarray(score['probabilities'],float)
    if score['id']!=request['id'] or score['option_ids']!=ids or p.shape!=(len(ids),):raise ValueError('Output contract mismatch')
    if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-6):raise ValueError('Bad probability simplex')
    if not np.isfinite(score['total_ms']) or score['total_ms']<=0:raise ValueError('Invalid measured latency')
    return ids[int(p.argmax())]

def run(model,shard,fixture,out):
    out.mkdir(parents=True,exist_ok=True);manifest=json.loads((fixture/'manifest.json').read_text())
    if sha(fixture/'requests.jsonl')!=manifest['requests_sha256']:raise ValueError('Input fixture changed')
    if (fixture/'gold.jsonl').exists():raise ValueError('Inference environment must not receive gold artifact')
    rows=read_jsonl(fixture/'requests.jsonl')
    if len(rows)!=320 or len({r['request']['id'] for r in rows})!=320:raise ValueError('Incomplete fixture')
    for r in rows:
        if digest(r['request'])!=r['sha256']:raise ValueError('Individual request changed')
    assigned=[r for r in rows if r['shard']==shard]
    if not assigned:raise ValueError('Empty shard')
    reverse=set();repeat=set()
    for ds in ('boolq','ocnli','clinc','tmmluplus'):
        test=[r for r in rows if r['dataset']==ds and r['role']=='test']
        reverse.update(r['request']['id'] for r in test[:4]);repeat.update(r['request']['id'] for r in test[:2])
    started=time.perf_counter();adapter=Adapter(model);loadms=1000*(time.perf_counter()-started)
    adapter.score(assigned[0]['request'])
    inventory=[]
    with (out/'predictions.jsonl').open('w') as stream:
        for r in assigned:
            for kind in ['primary']+(['reverse'] if r['request']['id'] in reverse else [])+(['repeat'] if r['request']['id'] in repeat else []):
                request=json.loads(json.dumps(r['request']))
                if kind=='reverse':request['options']=list(reversed(request['options']))
                score=adapter.score(request);validate_score(score,request)
                record={'dataset':r['dataset'],'role':r['role'],'group':r['group'],'kind':kind,'request_sha256':digest(request),'score':score}
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();inventory.append((request['id'],kind))
            print(model,shard,r['dataset'],r['role'],r['request']['id'],round(score['total_ms'],2),flush=True)
    write(out/'provenance.json',{'version':'1.6.0','commit':os.environ.get('GITHUB_SHA'),'model':model,'shard':shard,'fixture_sha256':manifest['requests_sha256'],'protocol_sha256':manifest['protocol_sha256'],'adapter':adapter.meta,'adapter_file_sha256':sha(Path(__file__).with_name('adapters.py')),'runner_sha256':sha(__file__),'forwards':len(inventory),'warmups':1,'primary':len(assigned),'load_and_download_ms':loadms,'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_model':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown'),'platform':platform.platform(),'run_url':os.environ.get('RESEARCH_RUN_URL'),'new_api_calls':0,'new_training_steps':0})
    write(out/'SHA256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256.json'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--fixture',type=Path,default=Path('fixture'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.model,a.shard,a.fixture,a.out)

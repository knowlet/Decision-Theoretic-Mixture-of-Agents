"""Canonical semantic fixtures, official source versions and disjoint groups.
Gold is a separate analysis artifact, never delivered to inference workers.
"""
from __future__ import annotations
import argparse,hashlib,json,time,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT=Path(__file__).resolve().parent
REVISIONS={'boolq':'35b264d03638db9f4ce671b711558bf7ff0f80d5','tmmluplus':'45e7c9b06d417c02a0998b2a6047383b04d608a9','ocnli':'b53efdee17257a5c33993cf6fcf8ffff0497ea0e','clinc':'828f8093932c8fe6ca7936c3d2e52903b1c523de'}
DATASETS=('boolq','ocnli','clinc','tmmluplus');SIZES={'fit':32,'dev':16,'test':32}
def canonical(obj):return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()
def group(text):return hashlib.sha256(' '.join(str(text).casefold().split()).encode()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def fetch(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DTMoA-1.6-research'}),timeout=120) as r:data=r.read()
                path.write_bytes(data);break
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
    return path

def download(cache):
    from huggingface_hub import snapshot_download
    paths={}
    for name,repo in [('boolq','google/boolq'),('tmmluplus','ikala/tmmluplus')]:
        paths[name]=Path(snapshot_download(repo,repo_type='dataset',revision=REVISIONS[name],allow_patterns=['*.parquet','*.csv','*.md'],local_dir=str(cache/name),max_workers=4))
    paths['ocnli']=cache/'ocnli'
    for split in ('train.50k','dev'):
        fetch(f"https://raw.githubusercontent.com/CLUEbenchmark/OCNLI/{REVISIONS['ocnli']}/data/ocnli/{split}.json",paths['ocnli']/f'{split}.json')
    fetch(f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISIONS['clinc']}/data/data_full.json",cache/'clinc/data_full.json');paths['clinc']=cache/'clinc'
    return paths

def row(ds,source_id,state,question,options,gold,group_text,source_split,subject=''):
    return {'dataset':ds,'source_id':str(source_id),'state':state,'question':question,'options':options,'gold':gold,'group':group(group_text),'source_split':source_split,'subject':subject}

def choose(rows,n,excluded=(),roundrobin=False):
    excluded=set(excluded);seen=set();ordered=sorted(rows,key=lambda r:digest(['jev16-fixed-cohort',r['group'],r['source_id']]))
    unique=[]
    for r in ordered:
        if r['group'] not in seen and r['group'] not in excluded:unique.append(r);seen.add(r['group'])
    if roundrobin:
        from collections import defaultdict
        queues=defaultdict(list)
        for r in unique:queues[r['subject']].append(r)
        subjects=sorted(queues,key=lambda s:digest(['jev16-subject',s]));unique=[]
        for i in range(max(map(len,queues.values()))):
            for s in subjects:
                if i<len(queues[s]):unique.append(queues[s][i])
    if len(unique)<n:raise ValueError('Insufficient distinct groups')
    return unique[:n]

def prepare(paths,out):
    out.mkdir(parents=True,exist_ok=True);selected=[];audits={};allrows={};sources=[]
    for ds,path in paths.items():
        for p in sorted(path.rglob('*')):
            if p.is_file() and '.cache' not in p.parts and p.suffix in ('.json','.csv','.parquet'):
                sources.append({'dataset':ds,'file':str(p.relative_to(path)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    options=[{'id':'no','text':'No'},{'id':'yes','text':'Yes'}];allrows['boolq']={}
    for src,role in [('train','train'),('validation','test')]:
        df=pd.concat([pd.read_parquet(p) for p in sorted(paths['boolq'].glob('data/'+src+'*.parquet'))])
        allrows['boolq'][role]=[row('boolq',f'{src}:{i}',r.passage,r.question,options,'yes' if r.answer else 'no',r.passage,src) for i,r in enumerate(df.itertuples())]
    allrows['ocnli']={};excluded_n=0
    opts=[{'id':'entailment','text':'蕴含：前提成立时，假设必然成立'},{'id':'neutral','text':'中立：根据前提无法确定假设是否成立'},{'id':'contradiction','text':'矛盾：前提成立时，假设必然不成立'}]
    for src,role in [('train.50k','train'),('dev','test')]:
        original=read_jsonl(paths['ocnli']/(src+'.json'));usable=[r for r in original if r['label'] in ('entailment','neutral','contradiction')];excluded_n+=len(original)-len(usable)
        allrows['ocnli'][role]=[row('ocnli',f'{src}:{r["id"]}',f'前提：{r["sentence1"]}\n假设：{r["sentence2"]}','根据前提，假设属于哪一种关系？',opts,r['label'],r['sentence1'],src,r.get('genre','')) for r in usable]
    audits['ocnli']={'unlabeled_or_no_majority_excluded':excluded_n,'evaluation':'official labeled dev, NOT blind official test','group':'premise text, one pair per selected premise'}
    allrows['tmmluplus']={}
    for suffix,role in [('dev','train'),('val','dev'),('test','test')]:
        rows=[]
        for path in sorted(paths['tmmluplus'].glob('data/*_'+suffix+'.csv')):
            subject=path.name.removesuffix('_'+suffix+'.csv');df=pd.read_csv(path,keep_default_na=False)
            for i,r in df.iterrows():
                opt=[{'id':c,'text':str(r[c])} for c in 'ABCD']
                if r['answer'] not in 'ABCD' or len(r['answer'])!=1:raise ValueError('Unexpected TMMLU label')
                rows.append(row('tmmluplus',f'{subject}:{suffix}:{i}',str(r['question']),'請選出此題的正確答案。',opt,r['answer'],str(r['question'])+'\n'+canonical(opt),suffix,subject))
        allrows['tmmluplus'][role]=rows
    cl=json.loads((paths['clinc']/'data_full.json').read_text());allrows['clinc']={}
    for src,role in [('train','train'),('val','dev'),('test','test')]:
        data=cl[src]+cl['oos_'+src]
        allrows['clinc'][role]=[row('clinc',f'{src}:{i}',text,'Which intent best describes this utterance?',[],lab,text,src) for i,(text,lab) in enumerate(data)]
    for ds in DATASETS:
        raw=allrows[ds];testgroup={r['group'] for r in raw['test']};rnd=ds=='tmmluplus'
        if ds=='clinc':
            test=choose([r for r in raw['test'] if r['gold']=='oos'],8)+choose([r for r in raw['test'] if r['gold']!='oos'],24)
            fit=choose([r for r in raw['train'] if r['gold']=='oos'],8,testgroup)+choose([r for r in raw['train'] if r['gold']!='oos'],24,testgroup)
            dev=choose([r for r in raw['dev'] if r['gold']=='oos'],4,testgroup|{r['group'] for r in fit})+choose([r for r in raw['dev'] if r['gold']!='oos'],12,testgroup|{r['group'] for r in fit})
        else:
            test=choose(raw['test'],SIZES['test'],roundrobin=rnd)
            fit=choose(raw['train'],SIZES['fit'],testgroup,roundrobin=rnd)
            dev=choose(raw.get('dev',raw['train']),SIZES['dev'],testgroup|{r['group'] for r in fit},roundrobin=rnd)
        for role,cohort in [('fit',fit),('dev',dev),('test',test)]:
            for r in cohort:r['role']=role;selected.append(r)
        audits.setdefault(ds,{}).update(source_rows={k:len(v) for k,v in raw.items()},selected={s:SIZES[s] for s in SIZES})
    clselected=[r for r in selected if r['dataset']=='clinc'];excluded={r['group'] for r in clselected}
    retrieval=[r for r in allrows['clinc']['train'] if r['gold']!='oos' and r['group'] not in excluded]
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True,norm='l2')
    matrix=vec.fit_transform([r['state'] for r in retrieval]);labels=sorted({r['gold'] for r in retrieval})
    if len(labels)!=150:raise ValueError('All 150 training intents required')
    bylabel={lab:np.array([i for i,r in enumerate(retrieval) if r['gold']==lab]) for lab in labels}
    for r in clselected:
        start=time.perf_counter();sim=(matrix@vec.transform([r['state']]).T).toarray().ravel()
        ranking=sorted(labels,key=lambda lab:(-float(sim[bylabel[lab]].max()),lab))[:8]
        r['retrieval_ms']=1000*(time.perf_counter()-start)
        ids=sorted(ranking+['oos'],key=lambda label:digest(['jev16-options',r['group'],label]))
        r['options']=[{'id':x,'text':('Out of scope: no supported intent' if x=='oos' else x.replace('_',' '))} for x in ids]
        r['question']='Select the utterance intent from the retrieved candidates, or out of scope when no supported intent applies.'
    audits['clinc'].update(retrieval_training_rows=len(retrieval),retrieval_intents=len(labels),shortlist=8,gold_forced_into_shortlist=False,scope='Retrieval-assisted full-intent classification, NOT an oracle 9-way classification. Missing gold counts as an error. OOS proportion deliberately 25%.')
    public=[];gold=[]
    for i,r in enumerate(sorted(selected,key=lambda r:(r['dataset'],r['role'],r['group']))):
        qid=f"{r['dataset']}:{r['group'][:20]}";req={'id':qid,'state':r['state'],'question':r['question'],'options':r['options']};ids=[o['id'] for o in req['options']]
        public.append({'request':req,'sha256':digest(req),'dataset':r['dataset'],'role':r['role'],'group':r['group'],'shard':i%12})
        gold.append({'id':qid,'dataset':r['dataset'],'role':r['role'],'group':r['group'],'source_id':r['source_id'],'source_split':r['source_split'],'subject':r['subject'],'label':r['gold'],'gold_index':ids.index(r['gold']) if r['gold'] in ids else len(ids),'gold_in_candidates':r['gold'] in ids,'option_ids':ids,'retrieval_ms':r.get('retrieval_ms',0.)})
    if len(public)!=320 or len({r['group'] for r in public})!=320:raise ValueError('Fixture group overlap')
    for ds in DATASETS:
        for role,n in SIZES.items():assert sum(r['dataset']==ds and r['role']==role for r in public)==n
    (out/'requests.jsonl').write_text(''.join(canonical(r)+'\n' for r in public));(out/'gold.jsonl').write_text(''.join(canonical(r)+'\n' for r in gold))
    write(out/'manifest.json',{'version':'1.6.0','revisions':REVISIONS,'audit':audits,'sources':sources,'cases':len(public),'requests_sha256':hashlib.sha256((out/'requests.jsonl').read_bytes()).hexdigest(),'gold_sha256':hashlib.sha256((out/'gold.jsonl').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'pretraining_contamination_excluded':False})
    print(json.dumps(audits,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('.cache/jev16'));p.add_argument('--out',type=Path,default=Path('fixture'));a=p.parse_args();prepare(download(a.cache),a.out)

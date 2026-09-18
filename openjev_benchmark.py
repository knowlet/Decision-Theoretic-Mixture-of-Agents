"""Actual upstream OpenJev readout versus matched empirical selectors.

Workers are archived; the selector is newly inferred locally on Actions CPU.
CPU loading differs from upstream's CUDA-only loader. direct.score is unmodified.
No target-test gold, unacquired answer, or gold-derived feature enters the scorer.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import transfer_study as t

ROOT=Path(__file__).resolve().parent
UPSTREAM='b4782a6c953f05c6255706d7a219f4e032af5b58'
MODELS={
 'qwen35-4b':('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
 'qwen35-0.8b':('Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17')}
DATASETS=t.PRIMARY_DATASETS
TEST_N,DEV_N,ORDER_N,REPEAT_N=32,16,8,2
PRICE,DEFER=.01,.25

def hash_object(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def select_subset(data,split,n):
    """Select unique exact groups by outcome-independent salted hash."""
    indices=np.flatnonzero(data.splits==split)
    order=sorted(indices,key=lambda i:hashlib.sha256(('openjev-h2h-v1:'+data.groups[i]).encode()).hexdigest())
    seen=set();selected=[]
    for i in order:
        if data.groups[i] in seen:continue
        seen.add(data.groups[i]);selected.append(i)
        if len(selected)==n:break
    if len(selected)!=n:raise ValueError('Insufficient independent groups')
    return data.subset(np.array(selected))

def render_request(case,reverse=False):
    """Only authorized public features; training-derived risks are not gold."""
    panel=case['panel'];answers=case['display_answers'];world_risks=case['estimated_errors']
    candidates=[{'candidate':int(i),'answer':answers[str(i)],'estimated_error_from_calibration':float(world_risks[str(i)])} for i in panel]
    options=[{'id':'emit_'+str(i),'description':'Return candidate '+str(i)+': '+answers[str(i)]} for i in panel]
    options.append({'id':'defer','description':'Do not return a candidate; defer to another decision maker at loss 0.25.'})
    rng=np.random.default_rng(int(case['group'][:8],16))
    order=rng.permutation(len(options));options=[options[i] for i in order]
    if reverse:options=list(reversed(options))
    row={'id':case['id']+(':reverse' if reverse else ''),
         'state':{'task':case['task'],'candidates':candidates,'costs':{'wrong_answer':1.,'correct_answer':0.,'defer':DEFER,'already_paid_acquisitions':3,'additional_acquisition_allowed':False},
                  'note':'Candidate errors are estimates from separate historical calibration data, not labels for this task. For multiple choice, the four original choices are ordered A, B, C, D.'},
         'question':'Choose the return or defer action with the lowest expected loss. Use the task evidence and the supplied estimated error rates. You may only return a listed candidate or defer. Do not generate a different answer.',
         'options':options}
    return row

def validate_result(result,request):
    ids=[x['id'] for x in request['options']];p=np.asarray(result['probabilities'],float)
    if result['option_ids']!=ids or p.shape!=(len(ids),) or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-6:
        raise ValueError('Invalid upstream probabilities/action set')
    return ids[int(np.argmax(p))]

def candidate_proposal(result):
    """Candidate proposal and confidence features, without consulting gold."""
    ids=result['option_ids'];p=np.asarray(result['probabilities'],float)
    indices=[i for i,x in enumerate(ids) if x.startswith('emit_')]
    i=max(indices,key=lambda i:p[i]);q=float(np.clip(p[i],1e-6,1-1e-6))
    entropy=float(-(p*np.log(np.maximum(p,1e-12))).sum()/np.log(len(p)))
    return int(ids[i].split('_')[1]),[math.log(q/(1-q)),entropy],q

class SuccessCalibrator:
    """Regularized Platt-like success calibration on dev, never test labels."""
    def fit(self,features,correct):
        x=np.asarray(features,float);y=np.asarray(correct,int)
        if len(x)!=len(y) or len(x)<8 or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('Bad calibration records')
        self.constant=None
        if len(set(y))==1:self.constant=float((y.sum()+1)/(len(y)+2));self.model=None
        else:self.model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=14).fit(x,y)
        return self
    def predict(self,x):
        x=np.asarray(x,float)
        return np.full(len(x),self.constant) if self.constant is not None else self.model.predict_proba(x)[:,1]
    def metadata(self):
        return {'constant':self.constant,'coef':None if self.model is None else self.model.coef_.tolist(),
                'intercept':None if self.model is None else self.model.intercept_.tolist(),'fit_split':'development only','regularization_C':1.}

def prepare(cache):
    protocol=json.loads((ROOT/'transfer_protocol.json').read_text())
    cases=[];baseline_frames=[];selection=[];resources=[]
    for ds in DATASETS:
        data,audit=t.load_dataset(cache/f'{ds}_predictions.csv',ds,protocol['source_sha256'][ds])
        tr=data.subset(data.splits=='train');dv=data.subset(data.splits=='dev')
        started=time.perf_counter();policies,head,recs=t.select(tr,dv,'replacement',PRICE);fit_seconds=time.perf_counter()-started
        selection.extend(recs);panel=policies['fixed3'].next_batch(t.ROOT_STATE)
        assert len(panel)==3
        for split,n in (('dev',DEV_N),('test',TEST_N)):
            part=select_subset(data,split,n)
            for r in range(len(part)):
                a=part.answers['replacement'][r];state=(sum(1<<i for i in panel),t.canonical(a[i] for i in sorted(panel)))
                errors={str(i):float(policies['fixed3'].world.risk[t.STATE_ID[state],i]) for i in panel}
                display={str(i):('UNPARSEABLE' if a[i] is None else str(a[i])) for i in panel}
                cases.append({'id':ds+':'+str(part.ids[r]),'dataset':ds,'sample_id':str(part.ids[r]),'group':str(part.groups[r]),'split':split,
                     'task':str(part.text[r]),'panel':list(map(int,panel)),'display_answers':display,'estimated_errors':errors,
                     'evaluation_errors':part.errors['replacement'][r].tolist()})
            if split=='test':
                risks=head.predict(part.text)
                for name,p in policies.items():
                    begun=time.perf_counter();f=t.evaluate(p,part,'replacement','headtohead',risks);seconds=time.perf_counter()-begun
                    f['algorithm']='empirical_'+name;f['controller_seconds']=seconds/len(f)
                    f['new_selector_calls']=0;f['input_tokens']=0;baseline_frames.append(f)
                p=policies['fixed3'];rows=[]
                for r in range(len(part)):
                    a=part.answers['replacement'][r];valid=[i for i in panel if a[i] is not None]
                    if valid:
                        chosen=min(valid,key=lambda i:(-sum(a[i]==a[j] for j in valid),p.world.global_error[i],i));err=float(part.errors['replacement'][r,chosen])
                    else:chosen=-1;err=DEFER
                    rows.append(dict(dataset=ds,sample_id=str(part.ids[r]),group=str(part.groups[r]),algorithm='matched_majority3',selected=chosen,answered=int(chosen>=0),loss=err,queries=3,cost=.03,objective=err+.03,controller_seconds=np.nan,new_selector_calls=0,input_tokens=0))
                baseline_frames.append(pd.DataFrame(rows))
        resources.append(dict(dataset=ds,train_n=len(tr),development_n=len(dv),historical_calibration_answers=4*(len(tr)+len(dv)),fit_select_seconds=fit_seconds,panel=list(map(int,panel))))
    return cases,pd.concat(baseline_frames,ignore_index=True),selection,resources

def cpu_model(label):
    import torch,transformers
    torch.set_num_threads(4)
    source,revision=MODELS[label];common={'revision':revision,'trust_remote_code':False};started=time.perf_counter()
    config=transformers.AutoConfig.from_pretrained(source,**common).get_text_config()
    tokenizer=transformers.AutoTokenizer.from_pretrained(source,**common)
    model,loading=transformers.Qwen3_5ForCausalLM.from_pretrained(source,config=config,dtype=torch.bfloat16,device_map={'':'cpu'},low_cpu_mem_usage=True,output_loading_info=True,**common)
    if any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):raise RuntimeError('Incomplete checkpoint: '+str(loading))
    model.eval()
    metadata={'source':source,'revision':revision,'dtype':'bfloat16','device':'cpu','threads':4,
        'torch':torch.__version__,'transformers':transformers.__version__,'upstream_commit':UPSTREAM,
        'loading_adapter':'CPU device instead of upstream CUDA-only loader; direct.score unchanged','model_load_seconds':time.perf_counter()-started}
    return model,tokenizer,metadata

def execute_benchmark(label,cache,out,vendor):
    out.mkdir(parents=True,exist_ok=True);vendor=vendor.resolve();sys.path.insert(0,str(vendor/'src'))
    from openjev_phase1.direct import score,encode_prompt
    import openjev_phase1.direct as upstream
    src=Path(upstream.__file__).resolve()
    if vendor not in src.parents:raise RuntimeError('Wrong upstream import')
    content=src.read_bytes();blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if blob!='943f34728d1966bf2325d6e6d34d2a7043cd6a56':raise RuntimeError('Upstream direct.py changed')
    cases,baselines,selection,resources=prepare(cache)
    baselines.to_csv(out/'baselines.csv',index=False);t.write_json(out/'selection.json',selection);t.write_json(out/'calibration_resources.json',resources)
    public_manifest=[{k:c[k] for k in ('id','dataset','sample_id','group','split','panel')}|{'request_sha256':hash_object(render_request(c))} for c in cases]
    t.write_json(out/'case_manifest.json',public_manifest)
    model,tokenizer,metadata=cpu_model(label)
    lengths=[len(encode_prompt(tokenizer,render_request(c),4096)[0]) for c in cases]
    print('prepared',len(cases),'input token range',min(lengths),max(lengths),flush=True)
    score(model,tokenizer,render_request(cases[0]),metadata,4096)
    outputs={}
    with (out/'selector_scores.jsonl').open('w') as dst:
        for j,c in enumerate(cases):
            request=render_request(c);r=score(model,tokenizer,request,metadata,4096);validate_result(r,request)
            outputs[c['id']]=r;dst.write(json.dumps(r,allow_nan=False)+'\n');dst.flush()
            if j%16==0:print('scored',j+1,'/',len(cases),flush=True)
    dev=[c for c in cases if c['split']=='dev'];features=[];correct=[]
    for c in dev:
        chosen,x,_=candidate_proposal(outputs[c['id']]);features.append(x);correct.append(1-c['evaluation_errors'][chosen])
    calibrator=SuccessCalibrator().fit(features,correct)
    t.write_json(out/'success_calibrator.json',calibrator.metadata()|{'n_development':len(dev),'development_ids':[c['id'] for c in dev]})
    rows=[];confidence_rows=[]
    for c in cases:
        if c['split']!='test':continue
        r=outputs[c['id']];candidate,x,rawq=candidate_proposal(r);pcal=float(calibrator.predict([x])[0])
        action=validate_result(r,render_request(c));rawchosen=-1 if action=='defer' else int(action.split('_')[1])
        confidence_rows.append(dict(id=c['id'],dataset=c['dataset'],group=c['group'],candidate=candidate,correct=1-c['evaluation_errors'][candidate],raw_option_score=rawq,calibrated_success=pcal))
        for algorithm,selected in [('openjev_raw_fixed3',rawchosen),('openjev_calibrated_fixed3',candidate if pcal>=1-DEFER else -1)]:
            if selected not in [-1,*c['panel']]:raise AssertionError('Unauthorized candidate')
            err=DEFER if selected<0 else float(c['evaluation_errors'][selected])
            rows.append(dict(dataset=c['dataset'],sample_id=c['sample_id'],group=c['group'],algorithm=algorithm,selected=selected,answered=int(selected>=0),loss=err,queries=3,cost=.03,objective=err+.03,
                controller_seconds=r['total_seconds'],new_selector_calls=1,input_tokens=r['input_tokens']))
    df=pd.concat([baselines,pd.DataFrame(rows)],ignore_index=True);df['selector_model']=label
    df.to_csv(out/'per_case.csv',index=False);pd.DataFrame(confidence_rows).to_csv(out/'confidence.csv',index=False)
    reverse=[];repeat=[]
    for ds in DATASETS:
        tests=[c for c in cases if c['dataset']==ds and c['split']=='test']
        for c in tests[:ORDER_N]:
            rr=score(model,tokenizer,render_request(c,True),metadata,4096)
            a=validate_result(outputs[c['id']],render_request(c));bb=validate_result(rr,render_request(c,True))
            old=dict(zip(outputs[c['id']]['option_ids'],outputs[c['id']]['probabilities']));new=dict(zip(rr['option_ids'],rr['probabilities']))
            reverse.append(dict(id=c['id'],dataset=ds,original=a,reversed=bb,flipped=int(a!=bb),tv=.5*sum(abs(old[k]-new[k]) for k in old),forward_seconds=rr['total_seconds']))
        for c in tests[:REPEAT_N]:
            rr=score(model,tokenizer,render_request(c),metadata,4096);old=outputs[c['id']]
            difference=float(np.max(np.abs(np.array(rr['probabilities'])-old['probabilities'])))
            repeat.append(dict(id=c['id'],max_probability_delta=difference,same_argmax=validate_result(rr,render_request(c))==validate_result(old,render_request(c))))
    pd.DataFrame(reverse).to_csv(out/'order_robustness.csv',index=False);t.write_json(out/'repeat_check.json',repeat)
    if not all(r['same_argmax'] and r['max_probability_delta']<1e-5 for r in repeat):raise AssertionError('Repeated readout unstable')
    t.write_json(out/'provenance.json',{'version':'1.5.0','tested_commit':os.getenv('GITHUB_SHA','local-uncommitted'),'selector_model':label,'metadata':metadata,
        'request_spec_sha256':t.digest(ROOT/'openjev_protocol.json'),'upstream_direct_git_blob':blob,
        'n_datasets':len(DATASETS),'n_test':len(DATASETS)*TEST_N,'n_development':len(DATASETS)*DEV_N,'new_primary_local_forwards':len(cases),'new_robustness_local_forwards':len(reverse)+len(repeat),'warmup_forwards':1,
        'new_llm_api_calls':0,'jev_proprietary_calls':0,'cera_training_steps':0,'worker_generation_calls':0,
        'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu':platform.processor(),'platform':platform.platform(),'python':sys.version,
        'cpu_model_name':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown'),
        'scope':'New local selector inference over immutable archived worker answers; neither live end-to-end MoA nor proprietary Jev. Small balanced pilot with previously available benchmark tasks.',
        'run_url':os.getenv('RESEARCH_RUN_URL')})
    t.write_json(out/'SHA256.json',{p.name:t.digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256.json'})
    print(df.groupby('algorithm').agg(objective=('objective','mean'),coverage=('answered','mean'),queries=('queries','mean')).to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--cache',type=Path,default=ROOT/'.cache/proeval');p.add_argument('--out',type=Path,required=True);p.add_argument('--vendor',type=Path,default=ROOT/'vendor/openjev');p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download:t.download(a.cache)
    execute_benchmark(a.model,a.cache,a.out,a.vendor)

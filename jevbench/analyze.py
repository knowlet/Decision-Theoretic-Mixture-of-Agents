"""Native-head benchmarks and calibrated acquisition ablations.
Policy latency is component replay, not freshly measured integrated serving.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from .fixtures import DATASETS,read_jsonl,digest,write
from .adapters import MODELS,sha
from .control import JointWorld,AgreementWorld,Compiled,observations
from .run import validate_score
NAMES=tuple(MODELS)

def load(incoming,fixture,evaluation):
    manifest=json.loads((fixture/'manifest.json').read_text());requests=read_jsonl(fixture/'requests.jsonl');gold=read_jsonl(evaluation/'gold.jsonl')
    assert sha(fixture/'requests.jsonl')==manifest['requests_sha256'] and sha(evaluation/'gold.jsonl')==manifest['gold_sha256']
    qmap={r['request']['id']:r for r in requests};gmap={r['id']:r for r in gold}
    assert set(qmap)==set(gmap) and len(qmap)==320 and len({r['group'] for r in requests})==320
    scores={name:{} for name in NAMES};probes=[];provenance=[];native=[]
    for shard in range(12):
        directories=list(incoming.glob(f'jev16-shard-{shard}-*'))
        if len(directories)!=1:raise ValueError('Shard unavailable or duplicated')
        cpus=[]
        for name in NAMES:
            root=directories[0]/'results'/name;checks=json.loads((root/'SHA256.json').read_text())
            for path,h in checks.items():
                if Path(path).name!=path or sha(root/path)!=h:raise ValueError('Bad inference artifact digest')
            p=json.loads((root/'provenance.json').read_text());provenance.append(p);cpus.append(p['cpu_model'])
            assert p['model']==name and p['shard']==shard and p['fixture_sha256']==manifest['requests_sha256']
            assert p['adapter']['id']==MODELS[name][0] and p['adapter']['revision']==MODELS[name][1]
            assert p['new_api_calls']==p['new_training_steps']==0
            records=read_jsonl(root/'predictions.jsonl');assert len(records)==p['forwards']
            for record in records:
                s=record['score'];qid=s['id'];kind=record['kind'];q=qmap[qid];g=gmap[qid]
                assert q['shard']==shard and q['role']==record['role'] and q['dataset']==record['dataset']
                req=json.loads(json.dumps(q['request']))
                if kind=='reverse':req['options'].reverse()
                assert digest(req)==record['request_sha256'];answer=validate_score(s,req)
                if kind=='primary':
                    if qid in scores[name]:raise ValueError('Duplicate primary inference')
                    scores[name][qid]=s
                    native.append({'model':name,'dataset':q['dataset'],'role':q['role'],'id':qid,'group':q['group'],'shard':shard,'correct':int(answer==g['label']),'prediction':answer,'label':g['label'],'latency_ms':s['total_ms'],'preprocess_ms':s['preprocess_ms'],'tokens_sum':s['input_tokens_sum'],'max_path_tokens':s['max_path_tokens'],'candidate_paths':s['candidate_paths'],'cpu_model':p['cpu_model'],'retrieval_ms':g['retrieval_ms']})
                elif kind in ('reverse','repeat'):probes.append((name,qid,kind,s))
                else:raise ValueError('Unexpected evaluation kind')
        assert len(set(cpus))==1
    assert all(set(scores[n])==set(qmap) for n in NAMES)
    assert sum(p['primary'] for p in provenance)==960 and sum(p['warmups'] for p in provenance)==36 and sum(p['forwards'] for p in provenance)==1032
    probe_rows=[]
    for name,qid,kind,s in probes:
        base=scores[name][qid];old=dict(zip(base['option_ids'],base['probabilities']));new=dict(zip(s['option_ids'],s['probabilities']));assert set(old)==set(new)
        difference=max(abs(old[x]-new[x]) for x in old);oldchoice=max(old,key=old.get);newchoice=max(new,key=new.get)
        probe_rows.append(dict(model=name,dataset=qmap[qid]['dataset'],id=qid,kind=kind,changed=oldchoice!=newchoice,max_probability_difference=difference,tv=.5*sum(abs(old[x]-new[x]) for x in old),same_host=True))
    probes=pd.DataFrame(probe_rows);assert len(probes)==72
    return manifest,requests,gold,scores,pd.DataFrame(native),probes,provenance

def wilson(correct,n):
    z=norm.ppf(.975);p=correct/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return center-width,center+width

def metrics(p,y):
    p=np.asarray(p,float);y=np.asarray(y,int)
    if y.max()>=p.shape[1]:p=np.pad(p,((0,0),(0,1)))
    truth=np.eye(p.shape[1])[y];top=p.max(1);pred=p.argmax(1);correct=pred==y;bins=np.minimum((top*10).astype(int),9)
    ece=sum(np.mean(bins==i)*abs(top[bins==i].mean()-correct[bins==i].mean()) for i in range(10) if np.any(bins==i))
    return dict(brier=float(((p-truth)**2).sum(1).mean()),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),ece=float(ece))

def make_frame(pred,queries,risk,indices,rows,y,latencies,algorithm,defer,seconds):
    records=[]
    for i,j in enumerate(indices):
        lab=int(pred[i]);used=tuple(map(int,queries[i]));g=rows[j];loss=defer if lab<0 else float(lab!=y[j])
        if len(set(used))!=len(used) or any(x not in range(3) for x in used) or len(used)>3:raise AssertionError('Invalid acquisition ledger')
        records.append(dict(dataset=g['dataset'],id=g['id'],group=g['group'],algorithm=algorithm,defer=defer,prediction=lab,gold=int(y[j]),answered=int(lab>=0),correct=int(lab>=0 and lab==y[j]),loss=loss,queries=len(used),query_order=','.join(map(str,used)),cost=.01*len(used),objective=loss+.01*len(used),reconstructed_ms=float(latencies[j,list(used)].sum()+g['retrieval_ms']+1000*seconds/len(indices)),controller_ms=1000*seconds/len(indices),estimated_risk=float(risk[i])))
    return pd.DataFrame(records)

def evaluate(policy,obs,indices,rows,y,timing,algorithm,defer,agreement=False):
    started=time.perf_counter();answers=[policy.execute(lambda model:int(obs[i,model]),defer=defer) if agreement else policy.execute(lambda model:int(obs[i,model])) for i in indices];elapsed=time.perf_counter()-started
    return make_frame([r[0] for r in answers],[r[1] for r in answers],[r[2] for r in answers],indices,rows,y,timing,algorithm,defer,elapsed)

def objective(policy,obs,y,indices,agreement=False,defer=.25):
    if not agreement:defer=policy.defer
    values=[]
    for i in indices:
        pred,used,_=policy.execute(lambda m:int(obs[i,m]),defer=defer) if agreement else policy.execute(lambda m:int(obs[i,m]))
        values.append((defer if pred<0 else float(pred!=y[i]))+.01*len(used))
    return float(np.mean(values))

def controls(requests,gold,scores,out):
    frames=[];selection=[];execution=[];native_metrics=[];oos=[];(out/'policies').mkdir(parents=True,exist_ok=True)
    for ds in DATASETS:
        rows=[g for g in gold if g['dataset']==ds];ids=[g['id'] for g in rows]
        p=np.array([[scores[name][qid]['probabilities'] for name in NAMES] for qid in ids]);y=np.array([g['gold_index'] for g in rows]);roles=np.array([g['role'] for g in rows]);timing=np.array([[scores[name][qid]['total_ms'] for name in NAMES] for qid in ids]);obs2=observations(p);obs1=observations(p,bins=1)
        fit=np.flatnonzero(roles=='fit');dev=np.flatnonzero(roles=='dev');test=np.flatnonzero(roles=='test');assert (len(fit),len(dev),len(test))==(32,16,32)
        for m,name in enumerate(NAMES):native_metrics.append(dict(dataset=ds,model=name,**metrics(p[test,m,:],y[test])))
        single=int(np.argmin((p[fit].argmax(-1)!=y[fit,None]).mean(0)));blind=AgreementWorld(p[fit],y[fit]);blindobs=p.argmax(-1)
        contract={'dataset':ds,'models':MODELS,'request_ids':ids,'version':'1.6.0','request_hashes':{r['request']['id']:r['sha256'] for r in requests if r['dataset']==ds},'fit_sha256':digest({'probs':p[fit].tolist(),'labels':y[fit].tolist()}),'controller_sha256':sha(Path(__file__).with_name('control.py'))}
        for defer in (.25,1.):
            pred=p[test,single,:].argmax(-1);singleframe=make_frame(pred,[(single,)]*len(test),np.zeros(len(test)),test,rows,y,timing,'selected_single',defer,0.);frames.append(singleframe)
            for method in ('majority','soft_vote'):
                start=time.perf_counter();pred=(np.array([np.bincount(v,minlength=p.shape[2]).argmax() for v in obs1[test]]) if method=='majority' else p[test].mean(1).argmax(1));elapsed=time.perf_counter()-start
                frames.append(make_frame(pred,[(0,1,2)]*len(test),np.zeros(len(test)),test,rows,y,timing,method,defer,elapsed))
            frames.append(evaluate(blind,blindobs,test,rows,y,timing,'agreement_only',defer,True))
            for name,bins,mode,grid in [('direction_only',1,'bellman',(8.,32.,128.)),('joint_bellman',2,'bellman',(8.,32.,128.)),('joint_myopic',2,'myopic',(8.,32.,128.)),('independent',2,'bellman',(float('inf'),))]:
                choices=[];start=time.perf_counter()
                for strength in grid:
                    world=JointWorld(p[fit],y[fit],bins=bins,strength=strength,missing=ds=='clinc');policy=world.compile(defer=defer,mode=mode)
                    score=objective(policy,obs1 if bins==1 else obs2,y,dev);choices.append((score,strength,policy))
                devloss,strength,policy=min(choices,key=lambda r:r[:2]);fitms=1000*(time.perf_counter()-start)
                selection.append(dict(dataset=ds,algorithm=name,defer=defer,strength=strength,dev_objective=devloss,fit_select_ms=fitms,fit_cases=len(fit),dev_cases=len(dev)))
                frame=evaluate(policy,obs1 if bins==1 else obs2,test,rows,y,timing,name,defer);frames.append(frame)
                file=out/'policies'/f'{ds}-{name}-{defer}.npz';version=contract|{'objective':defer,'policy':name};policy.save(file,version)
                if name=='joint_bellman':
                    loaded=Compiled.load(file,version)
                    for j in test:
                        expected=policy.execute(lambda m:int(obs2[j,m]));assert expected==loaded.execute(lambda m:int(obs2[j,m]))==policy.execute(lambda m:int(obs2[j,m]),recompute=True)
                    durations={}
                    for recompute in (False,True):
                        measurements=[]
                        for repeat in range(5):
                            start=time.perf_counter()
                            for j in test:
                                for _ in range(10):policy.execute(lambda m:int(obs2[j,m]),recompute=recompute)
                            measurements.append(1000*(time.perf_counter()-start)/(10*len(test)))
                        durations['online_q_ms' if recompute else 'compiled_ms']=float(np.median(measurements))
                    execution.append(dict(dataset=ds,defer=defer,exact_trace_match=True,states=len(policy.action),bytes=sum(a.nbytes for a in (policy.action,policy.choice,policy.terminal,policy.value,policy.mass)),**durations))
                    singledev=float(np.mean((p[dev,single].argmax(1)!=y[dev]).astype(float)+.01))
                    gated=frame.copy() if devloss<=singledev-.01 else singleframe.copy();gated['algorithm']='development_gated';frames.append(gated)
                    selection.append(dict(dataset=ds,algorithm='development_gated',defer=defer,strength=strength,dev_objective=min(devloss,singledev),selected='joint_bellman' if devloss<=singledev-.01 else 'selected_single',single_dev=singledev,gate_margin=.01))
            w=JointWorld(p[fit],y[fit]);pol=w.compile(defer=defer);pred=[];risk=[]
            for j in test:
                state=[0,0,0];state[single]=int(obs2[j,single])+1;index=pol.index(state);pred.append(int(pol.choice[index]));risk.append(pol.terminal[index])
            frames.append(make_frame(pred,[(single,)]*len(test),risk,test,rows,y,timing,'calibrated_single',defer,0.))
        if ds=='clinc':
            for m,name in enumerate(NAMES):
                inscope=np.array([rows[j]['label']!='oos' for j in test]);choice=p[test,m].argmax(1);oospred=np.array([rows[j]['option_ids'][choice[i]]=='oos' for i,j in enumerate(test)])
                oos.append(dict(model=name,n=len(test),in_scope=int(inscope.sum()),oos=int((~inscope).sum()),retrieval_recall=float(np.mean([rows[j]['gold_in_candidates'] for j in test if rows[j]['label']!='oos'])),oos_recall=float(oospred[~inscope].mean()),false_oos_rate=float(oospred[inscope].mean())))
    frame=pd.concat(frames,ignore_index=True);frame.to_csv(out/'policy_cases.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'selection.csv',index=False);pd.DataFrame(execution).to_csv(out/'compiled_latency.csv',index=False);pd.DataFrame(native_metrics).to_csv(out/'native_calibration.csv',index=False);pd.DataFrame(oos).to_csv(out/'clinc_oos.csv',index=False)
    return frame

def summarize_policy(frame):
    rows=[]
    for (ds,alg,defer),g in frame.groupby(['dataset','algorithm','defer']):
        covered=int(g.answered.sum());rows.append(dict(dataset=ds,algorithm=alg,defer=defer,n=len(g),objective=g.objective.mean(),coverage=g.answered.mean(),accuracy_all=g.correct.mean(),accuracy_answered=float(g.correct.sum()/covered) if covered else np.nan,queries=g.queries.mean(),reconstructed_p50_ms=g.reconstructed_ms.median(),reconstructed_p95_ms=g.reconstructed_ms.quantile(.95)))
    return pd.DataFrame(rows)

def comparisons(frame):
    pairs=[('joint_bellman','agreement_only',.25),('joint_bellman','selected_single',.25),('development_gated','selected_single',.25),('joint_bellman','agreement_only',1.)];rows=[]
    for aa,bb,defer in pairs:
        samples=[];points=[]
        for index,ds in enumerate(DATASETS):
            x=frame[(frame.dataset==ds)&(frame.algorithm==aa)&(frame.defer==defer)].set_index('id');z=frame[(frame.dataset==ds)&(frame.algorithm==bb)&(frame.defer==defer)].set_index('id').reindex(x.index)
            assert len(x)==32 and not z.objective.isna().any();diff=(x.objective-z.objective).to_numpy();rng=np.random.default_rng(np.random.SeedSequence([160918,index]));boot=diff[rng.integers(0,32,(2000,32))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(dataset=ds,a=aa,b=bb,defer=defer,difference=diff.mean(),lo=lo,hi=hi));samples.append(boot);points.append(diff.mean())
        boot=np.mean(samples,axis=0);lo,hi=np.quantile(boot,[.025,.975]);flo,fhi=np.quantile(boot,[.00625,.99375]);rows.append(dict(dataset='macro',a=aa,b=bb,defer=defer,difference=np.mean(points),lo=lo,hi=hi,family_lo=flo,family_hi=fhi))
    return pd.DataFrame(rows)

def run(incoming,fixture,evaluation,out):
    out.mkdir(parents=True,exist_ok=True);manifest,requests,gold,scores,native,probes,provenance=load(incoming,fixture,evaluation)
    native.to_csv(out/'native_cases.csv',index=False);probes.to_csv(out/'robustness_cases.csv',index=False)
    frame=controls(requests,gold,scores,out);pol=summarize_policy(frame);pol.to_csv(out/'policy_summary.csv',index=False);comparisons(frame).to_csv(out/'comparisons.csv',index=False)
    rows=[]
    for (ds,model),g in native[native.role=='test'].groupby(['dataset','model']):
        lo,hi=wilson(g.correct.sum(),len(g));rows.append(dict(dataset=ds,model=model,n=len(g),correct=int(g.correct.sum()),accuracy=g.correct.mean(),wilson_low=lo,wilson_high=hi,p50_ms=g.latency_ms.median(),p95_ms=g.latency_ms.quantile(.95),tokens_sum_mean=g.tokens_sum.mean(),max_path_tokens=int(g.max_path_tokens.max()),candidate_paths=int(g.candidate_paths.iloc[0]),paired_hosts=len(set(g.shard))))
    summary=pd.DataFrame(rows);summary.to_csv(out/'native_summary.csv',index=False)
    assert not frame.duplicated(['dataset','id','algorithm','defer']).any();np.testing.assert_allclose(frame.objective,frame.loss+.01*frame.queries,atol=1e-12,rtol=0)
    assert frame.queries.between(0,3).all() and ((frame.prediction==-1)==(frame.answered==0)).all()
    for _,g in frame.groupby(['dataset','algorithm','defer']):assert len(g)==32 and len(set(g.group))==32
    write(out/'verification.json',{'version':'1.6.0','all_gates_passed':True,'inference_commits':sorted({p['commit'] for p in provenance}),'analysis_commit':os.getenv('GITHUB_SHA'),'inference_run_urls':sorted({p['run_url'] for p in provenance}),'analysis_run_url':os.getenv('RESEARCH_RUN_URL'),'models':MODELS,'source_manifest':manifest,'native_predictions':len(native),'test_predictions':int((native.role=='test').sum()),'primary_forwards':960,'probe_forwards':72,'warmup_forwards':36,'total_forwards':1068,'native_loading_provenance':provenance,'same_host_pairing':True,'same_semantic_request_hashes':True,'new_api_calls':0,'model_training_steps':0,'scope':'Pilot: 32 test groups per task. Native head evaluation; policy latency is reconstructed, not integrated serving measurement. Fitted-policy optimization does not assert a held-out win.'})
    print(summary.to_string(index=False));print(pol.groupby(['algorithm','defer']).objective.mean().to_string())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--incoming',type=Path,default=Path('incoming'));p.add_argument('--fixture',type=Path,default=Path('fixture'));p.add_argument('--evaluation',type=Path,default=Path('evaluation'));p.add_argument('--out',type=Path,default=Path('results/jev16'));a=p.parse_args();run(a.incoming,a.fixture,a.evaluation,a.out)

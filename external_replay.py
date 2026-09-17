"""Honest offline replay of historical LLM responses; no live inference.

The policy receives only an acquisition callback. Gold/quality labels and
unqueried prices exist exclusively in the evaluator. Never import the
RouterBench oracle routing column. Cost fields are historical estimates.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib, itertools, json, pickle, platform
import re, sys, time, unicodedata, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
MODELS = ('mistralai/mixtral-8x7b-chat', 'gpt-3.5-turbo-1106',
          'claude-v2', 'gpt-4-1106-preview')
REVISION = '784021482c3f320c6619ed4b3bb3b41a21424fcb'
SOURCE_SHA = 'ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d'
FAMILIES = ('mmlu', 'arc-challenge', 'hellaswag', 'winogrande')
POLICIES = ('single_best', 'fixed_three', 'static_all_sources', 'myopic',
            'disagreement_gate', 'majority_three', 'defer_all', 'bellman')
N, K, BUDGET = 4, 5, 3  # four answers plus unparsed/INVALID, not correctness bits
POW = 6 ** np.arange(N)
STATES = (np.arange(6**N)[:, None] // POW) % 6
FULL = np.array(list(itertools.product(range(K), repeat=N)), dtype=np.int8)
FULL_ID_POW = K ** np.arange(N-1, -1, -1)
MATCH = np.all((STATES[:, None, :] == 0) |
               (STATES[:, None, :] == FULL[None, :, :] + 1), axis=2).astype(float)
OBSERVED = STATES > 0
COUNTS = OBSERVED.sum(axis=1)
MASKS = np.arange(1 << N)
MASK_BITS = (MASKS[:, None] >> np.arange(N)) & 1
TRANS = {}
for s in range(len(STATES)):
    seen = sum(1 << i for i in range(N) if STATES[s,i])
    options = []
    for m in range(1, 1 << N):
        if m & seen or m.bit_count() > BUDGET - COUNTS[s]:
            continue
        ix = np.flatnonzero(MASK_BITS[m])
        children = s + np.unique((FULL[:,ix] + 1) @ POW[ix])
        options.append((m, children))
    TRANS[s] = options
ORDER = sorted(range(len(STATES)), key=lambda s: int(COUNTS[s]), reverse=True)

# Load this specific hash-pinned upstream pickle using a strict global allowlist.
# No generated code or pickle-specified arbitrary functions are executed.
ALLOWED = {('numpy','dtype'), ('numpy','ndarray'),
 ('pandas.core.internals.managers','BlockManager'), ('pandas.core.frame','DataFrame'),
 ('pandas._libs.internals','_unpickle_block'), ('pandas.core.indexes.base','_new_Index'),
 ('pandas.core.indexes.base','Index'), ('pandas.core.indexes.range','RangeIndex'),
 ('numpy.core.numeric','_frombuffer'), ('numpy.core.multiarray','_reconstruct'),
 ('builtins','slice')}
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'Unapproved global: {module}.{name}')
        return getattr(importlib.import_module(module), name)

def load_source(path: Path) -> pd.DataFrame:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError(f'Source hash mismatch: {digest}')
    with path.open('rb') as f:
        return RestrictedUnpickler(f).load()

def download_source(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f'https://huggingface.co/datasets/withmartian/routerbench/resolve/{REVISION}/routerbench_0shot.pkl?download=true'
        tmp = path.with_suffix('.partial')
        urllib.request.urlretrieve(url, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != SOURCE_SHA:
            tmp.unlink(missing_ok=True)
            raise ValueError('Downloaded source hash mismatch')
        tmp.replace(path)

def unwrap(value) -> str:
    if isinstance(value, (list, tuple)):
        return '\n'.join(map(str,value))
    if not isinstance(value, str):
        return ''
    s = value.strip()
    if s.startswith('[') and len(s) < 1_000_000:
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list) and all(isinstance(x,str) for x in v):
                return '\n'.join(v)
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            pass
    return s

def parse_answer(value) -> int:
    """Strict response-only prefix parser; failures remain visible as category 4."""
    s = unwrap(value).strip()
    # Only an initial answer marker; never search all text for a gold-matching letter.
    m = re.match(r'^(?:[\s*`#]*)(?:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*)?[:=]?\s*)?[\(\[]?([A-D])(?=$|[\s\)\].,:;!])',s,re.I)
    return ord(m.group(1).upper()) - ord('A') if m else 4

def group_hash(prompt) -> str:
    normalized = ' '.join(unicodedata.normalize('NFKC',unwrap(prompt)).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()

def split_group(group: str) -> str:
    h = hashlib.sha256(('dtmoa-external-v1:' + group).encode()).digest()
    u = int.from_bytes(h[:8], 'big') / 2**64
    return 'train' if u < .6 else ('dev' if u < .8 else 'test')

@dataclass
class TraceData:
    ids: np.ndarray
    groups: np.ndarray
    family: np.ndarray
    subject: np.ndarray
    splits: np.ndarray
    answers: np.ndarray
    scores: np.ndarray
    costs: np.ndarray
    def subset(self, mask):
        return TraceData(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))
    def __len__(self):
        return len(self.ids)

def prepare(df: pd.DataFrame) -> tuple[TraceData, dict]:
    required = ['sample_id','prompt','eval_name'] + [c for m in MODELS for c in (m,m+'|model_response',m+'|total_cost')]
    missing = set(required) - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {missing}')
    # Explicit allowlist drops oracle_model_to_route_to before any processing.
    d = df.loc[:,required].copy()
    included = d.eval_name.astype(str).str.startswith('mmlu-') | d.eval_name.isin(FAMILIES[1:])
    d = d[included].copy()
    audit = {'source_rows':len(df),'eligible_rows':len(d),'excluded_outside_four_families':int((~included).sum())}
    scores = d[list(MODELS)].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    costs = d[[m+'|total_cost' for m in MODELS]].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = np.isfinite(scores).all(axis=1) & np.isin(scores,[0,1]).all(axis=1) & np.isfinite(costs).all(axis=1) & (costs>=0).all(axis=1)
    audit['excluded_missing_or_nonbinary_score_or_invalid_cost'] = int((~valid).sum())
    audit['excluded_by_eval'] = d.loc[~valid,'eval_name'].value_counts().to_dict()
    d = d[valid].copy(); scores=scores[valid]; costs=costs[valid]
    if d.sample_id.duplicated().any(): raise ValueError('Duplicate sample identifiers')
    groups = np.array([group_hash(p) for p in d.prompt])
    if any(not unwrap(p).strip() for p in d.prompt): raise ValueError('Empty prompt')
    answers = np.array([[parse_answer(v) for v in d[m+'|model_response']] for m in MODELS],dtype=np.int8).T
    families = np.array(['mmlu' if str(v).startswith('mmlu-') else v for v in d.eval_name])
    traces=TraceData(d.sample_id.astype(str).to_numpy(),groups,families,d.eval_name.astype(str).to_numpy(),
                     np.array([split_group(g) for g in groups]),answers,scores,costs)
    audit['retained_rows']=len(traces);audit['unique_prompt_groups']=len(set(groups))
    audit['duplicate_prompt_rows']=len(traces)-len(set(groups))
    audit['invalid_answer_count_by_model']=dict(zip(MODELS,(answers==4).sum(axis=0).tolist()))
    audit['split_counts']={s:int((traces.splits==s).sum()) for s in ('train','dev','test')}
    audit['family_counts']=dict(zip(*np.unique(families,return_counts=True)))
    # Exact prompt groups are assigned once, independent of model outcomes.
    group_sets=[set(groups[traces.splits==s]) for s in ('train','dev','test')]
    audit['cross_split_group_overlap']=sum(len(a&b) for a,b in itertools.combinations(group_sets,2))
    assert audit['cross_split_group_overlap']==0
    return traces,audit

@dataclass
class Fitted:
    mass: np.ndarray
    mean_scores: np.ndarray
    expected_cost: np.ndarray
    global_scores: np.ndarray
    joint_mass: np.ndarray
    joint_score_sums: np.ndarray

def fit(answers: np.ndarray, scores: np.ndarray, costs: np.ndarray, strength: float) -> Fitted:
    if answers.ndim!=2 or answers.shape[1]!=N or answers.shape!=scores.shape or scores.shape!=costs.shape or len(answers)==0:
        raise ValueError('Expected nonempty aligned n-by-four arrays')
    if not np.isin(answers,range(K)).all() or not np.isin(scores,[0,1]).all(): raise ValueError('Bad answer or score alphabet')
    if not np.isfinite(costs).all() or (costs<0).any() or not np.isfinite(strength) or strength<=0: raise ValueError('Invalid resource/prior')
    bins=answers@FULL_ID_POW
    count=np.bincount(bins,minlength=K**N).astype(float)
    sums=np.column_stack([np.bincount(bins,weights=scores[:,i],minlength=K**N) for i in range(N)])
    marg=np.column_stack([(np.bincount(answers[:,i],minlength=K)+.5)/(len(answers)+.5*K) for i in range(N)])
    prior=np.prod(np.column_stack([marg[FULL[:,i],i] for i in range(N)]),axis=1)
    global_score=(scores.sum(axis=0)+.5)/(len(scores)+1)
    joint=count+strength*prior
    joint_sums=sums+strength*prior[:,None]*global_score
    masses=MATCH@joint
    mean=(MATCH@joint_sums)/masses[:,None]
    return Fitted(masses,mean,costs.mean(axis=0),global_score,joint,joint_sums)

@dataclass
class ReplayPolicy:
    actions: np.ndarray
    choices: np.ndarray
    predicted_value: float
    name: str

def terminal(fitted: Fitted, defer_loss: float):
    risks=np.where(OBSERVED,1-fitted.mean_scores,np.inf)
    all_risks=np.column_stack([risks,np.full(len(STATES),defer_loss)])
    choices=np.argmin(all_risks,axis=1)
    risk=all_risks[np.arange(len(STATES)),choices]
    return np.where(choices==N,-1,choices).astype(np.int8),risk

def build_policy(fitted: Fitted, name: str, lam: float=10., defer_loss:float=.25) -> ReplayPolicy:
    if name not in POLICIES: raise ValueError(name)
    if min(lam,defer_loss)<0 or not np.isfinite([lam,defer_loss]).all(): raise ValueError('Invalid loss')
    choice,risk=terminal(fitted,defer_loss)
    actions=np.zeros(len(STATES),dtype=np.int8)
    charges=lam*(MASK_BITS@fitted.expected_cost)
    def expected(m,children,values):
        # Children partition parent support. Use their unnormalized masses.
        p=fitted.mass[children]; return charges[m]+np.dot(p,values[children])/p.sum()
    if name in ('bellman','myopic'):
        values=risk.copy()
        for s in ORDER:
            best=risk[s]
            for m,ch in TRANS[s]:
                if name=='myopic' and m.bit_count()!=1: continue
                q=expected(m,ch,values if name=='bellman' else risk)
                if q<best-1e-12: best=q; actions[s]=m
            values[s]=best
        return ReplayPolicy(actions,choice,float(values[0]),name)
    if name=='defer_all': return ReplayPolicy(actions,np.full_like(choice,-1),defer_loss,name)
    if name in ('single_best','fixed_three','static_all_sources'):
        sizes={'single_best':(1,), 'fixed_three':(3,), 'static_all_sources':(0,1,2,3)}[name]
        best=defer_loss if 0 in sizes else np.inf
        for m,ch in TRANS[0]:
            if m.bit_count() not in sizes: continue
            q=expected(m,ch,risk)
            if q<best-1e-12: best=q; actions[0]=m
        return ReplayPolicy(actions,choice,float(best),name)
    if name=='disagreement_gate':
        cheap=np.argsort(fitted.expected_cost,kind='stable')[:2]
        first=sum(1<<int(i) for i in cheap); actions[0]=first
        third=next(int(i) for i in np.argsort(-fitted.global_scores,kind='stable') if i not in cheap)
        for s in range(len(STATES)):
            if COUNTS[s]==2 and np.all(OBSERVED[s,cheap]):
                a,b=STATES[s,cheap]-1
                if a==4 or b==4 or a!=b: actions[s]=1<<third
        return ReplayPolicy(actions,choice,float('nan'),name)
    # Standard fixed best-three majority, no abstention (diagnostic reference).
    panel=np.argsort(-fitted.global_scores,kind='stable')[:3]
    actions[0]=sum(1<<int(i) for i in panel)
    for s in range(len(STATES)):
        seen=np.flatnonzero(OBSERVED[s]); valid=[int(i) for i in seen if STATES[s,i]!=5]
        if not len(seen): continue
        if valid:
            votes={i:sum(STATES[s,j]==STATES[s,i] for j in valid) for i in valid}
            choice[s]=max(valid,key=lambda i:(votes[i],fitted.global_scores[i],-i))
        else: choice[s]=max(seen,key=lambda i:(fitted.global_scores[i],-i))
    return ReplayPolicy(actions,choice,float('nan'),name)

def execute(policy: ReplayPolicy, acquire: Callable[[int],tuple[int,float]]):
    """Observation boundary: the callback returns ONLY answer category and paid cost."""
    s=0; total=0.; acquired=[]; batches=[]
    for _ in range(BUDGET+1):
        mask=int(policy.actions[s])
        if mask==0:
            selected=int(policy.choices[s])
            if selected>=0 and selected not in acquired: raise AssertionError('Unacquired candidate selected')
            return selected,total,len(acquired),s,tuple(batches)
        if mask<0 or mask>=1<<N: raise AssertionError('Invalid action mask')
        batch=np.flatnonzero(MASK_BITS[mask])
        if len(batch)+len(acquired)>BUDGET or any(int(i) in acquired for i in batch): raise AssertionError('Invalid acquisition')
        for i in batch:
            answer,cost=acquire(int(i))
            if answer not in range(K) or cost<0 or not np.isfinite(cost): raise ValueError('Bad observation')
            s+=int(POW[i]*(answer+1));total+=float(cost);acquired.append(int(i))
        batches.append(mask)
    raise AssertionError('Policy did not stop')

def evaluate(policy: ReplayPolicy, data: TraceData, lam:float, defer_loss:float, scenario:str) -> pd.DataFrame:
    rows=[]
    for row in range(len(data)):
        def acquire(i): return int(data.answers[row,i]),float(data.costs[row,i])
        selected,cost,n,s,batches=execute(policy,acquire)
        # Gold/quality is first consulted AFTER the final policy decision.
        score=float(data.scores[row,selected]) if selected>=0 else np.nan
        loss=1-score if selected>=0 else defer_loss
        rows.append((data.ids[row],data.groups[row],data.family[row],data.subject[row],scenario,policy.name,
                     lam,defer_loss,selected,int(selected>=0),score,loss,cost,loss+lam*cost,n,s,';'.join(map(str,batches))))
    return pd.DataFrame(rows,columns=['sample_id','prompt_group','family','subject','scenario','policy','lambda_usd','defer_loss',
                                      'selected_model_index','answered','selected_score','terminal_loss','historical_cost_usd',
                                      'objective','acquisitions','terminal_state','batch_masks'])

def summary(frame):
    out=[]
    for keys,g in frame.groupby(['scenario','lambda_usd','defer_loss','policy'],sort=True):
        out.append(dict(zip(['scenario','lambda_usd','defer_loss','policy'],keys),n=len(g),objective=g.objective.mean(),
           terminal_loss=g.terminal_loss.mean(),historical_cost_usd=g.historical_cost_usd.mean(),coverage=g.answered.mean(),
           selective_accuracy=g.selected_score.mean(),wrong_per_all=((g.answered==1)&(g.selected_score==0)).mean(),
           acquisitions=g.acquisitions.mean(),macro_family_objective=g.groupby('family').objective.mean().mean()))
    return pd.DataFrame(out)

def paired_bootstrap(frame, count=2000, seed=17092026):
    """Resample unique prompt groups, preserving pairs; conditional on fitted policies."""
    records=[]
    for base in ('myopic','static_all_sources','single_best'):
        a=frame[frame.policy=='bellman'].set_index('sample_id')
        b=frame[frame.policy==base].set_index('sample_id').reindex(a.index)
        if b.objective.isna().any(): raise ValueError('Unpaired evaluation')
        difference=(a.objective-b.objective).to_numpy()
        grouped=pd.DataFrame({'group':a.prompt_group.to_numpy(),'difference':difference}).groupby('group').difference.agg(['sum','count'])
        rng=np.random.default_rng(seed)
        sums=grouped['sum'].to_numpy();counts=grouped['count'].to_numpy();n=len(sums)
        means=[]
        for start in range(0,count,100):
            ix=rng.integers(0,n,size=(min(100,count-start),n))
            means.extend((sums[ix].sum(axis=1)/counts[ix].sum(axis=1)).tolist())
        lo,hi=np.quantile(means,[.025,.975])
        # Group-clustered t approximation for a paired mean; not a retraining CI.
        avg=float(difference.mean());residual=sums-avg*counts
        se=float(np.sqrt(n/(n-1)*np.sum(residual**2))/counts.sum()) if n>1 else np.nan
        p=float(2*stats.t.sf(abs(avg/se),n-1)) if se>0 else (1. if avg==0 else 0.)
        records.append({'a':'bellman','b':base,'difference':avg,'bootstrap_lo':lo,'bootstrap_hi':hi,
                        'p_cluster_t':p,'n_prompt_groups':n,'bootstrap_replicates':count})
    order=np.argsort([r['p_cluster_t'] for r in records]);last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(len(records)-rank)*records[index]['p_cluster_t']))
        records[index]['holm_p']=last
    return pd.DataFrame(records)

def train_select(train,dev,lam,defer):
    fits={strength:fit(train.answers,train.scores,train.costs,strength) for strength in (1.,10.,100.)}
    policies={};selection=[]
    for name in POLICIES:
        options=[]
        for strength,f in fits.items():
            p=build_policy(f,name,lam,defer)
            val=evaluate(p,dev,lam,defer,'development').objective.mean()
            options.append((float(val),strength,p))
        val,strength,p=min(options,key=lambda x:(x[0],x[1]))
        policies[name]=p
        selection.append({'policy':name,'prior_strength':strength,'dev_objective':val,'lambda_usd':lam,'defer_loss':defer})
    return policies,selection

def run(source:Path,output:Path,bootstrap:int=2000):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    data,audit=prepare(load_source(source))
    manifest=pd.DataFrame({k:getattr(data,k) for k in ('ids','groups','family','subject','splits')})
    manifest.to_csv(output/'split_manifest.csv',index=False)
    train=data.subset(data.splits=='train');dev=data.subset(data.splits=='dev');test=data.subset(data.splits=='test')
    primary=[];all_summary=[];selected=[];primary_policies=None
    for defer in (.25,1.):
        for lam in (0.,1.,10.,100.,1000.):
            policies,selections=train_select(train,dev,lam,defer)
            for x in selections:x['scenario']='in_distribution'
            selected.extend(selections)
            frames=[evaluate(p,test,lam,defer,'in_distribution') for p in policies.values()]
            frame=pd.concat(frames,ignore_index=True);all_summary.append(summary(frame))
            if lam==10. and defer==.25:
                primary=frame;primary_policies=policies
                frame.to_csv(output/'primary_per_case.csv',index=False)
            print('ID',lam,defer,flush=True)
    pd.concat(all_summary,ignore_index=True).to_csv(output/'cost_deferral_sweep.csv',index=False)
    summary(primary).to_csv(output/'primary_summary.csv',index=False)
    paired_bootstrap(primary,bootstrap).to_csv(output/'primary_comparisons.csv',index=False)
    primary.groupby(['family','policy']).agg(n=('sample_id','size'),objective=('objective','mean'),
        cost=('historical_cost_usd','mean'),coverage=('answered','mean'),selective_accuracy=('selected_score','mean')).reset_index().to_csv(output/'primary_by_family.csv',index=False)
    ood=[]
    for family in FAMILIES:
        tr=data.subset((data.splits=='train')&(data.family!=family));dv=data.subset((data.splits=='dev')&(data.family!=family))
        te=data.subset((data.splits=='test')&(data.family==family))
        policies,selections=train_select(tr,dv,10.,.25)
        for x in selections:x['scenario']='heldout_'+family
        selected.extend(selections)
        ood.extend(evaluate(p,te,10.,.25,'heldout_'+family) for p in policies.values())
        print('OOD',family,flush=True)
    ood_frame=pd.concat(ood,ignore_index=True)
    ood_frame.to_csv(output/'ood_per_case.csv',index=False)
    summary(ood_frame).to_csv(output/'ood_summary.csv',index=False)
    pd.DataFrame(selected).to_csv(output/'development_selection.csv',index=False)
    # Post-evaluation diagnostics; these are not used to choose policies.
    diagnostic=[]
    for family in FAMILIES:
        mask=test.family==family;an=test.answers[mask];sc=test.scores[mask]
        agree=(an[:,0]!=4)&np.all(an==an[:,[0]],axis=1)
        diagnostic.append({'family':family,'n':int(mask.sum()),'all_four_agree_n':int(agree.sum()),
           'all_four_agree_and_wrong_n':int((agree&np.all(sc==0,axis=1)).sum()),
           'all_four_wrong_given_agreement':float(np.all(sc[agree]==0,axis=1).mean()) if agree.any() else None,
           'any_selected_model_correct':float((sc.max(axis=1)>0).mean())})
    pd.DataFrame(diagnostic).to_csv(output/'common_error_diagnostics.csv',index=False)
    pd.DataFrame(np.corrcoef(1-test.scores,rowvar=False),index=MODELS,columns=MODELS).to_csv(output/'test_error_correlations.csv')
    # Store only derived numeric/categorical traces, not copyrighted prompt/answer text.
    np.savez_compressed(output/'derived_traces.npz',answers=data.answers,scores=data.scores,costs=data.costs,
                         sample_id=data.ids.astype(str),prompt_group=data.groups.astype(str),family=data.family.astype(str),splits=data.splits.astype(str))
    # Deterministic replay checks on exactly the held-out primary policies.
    repeated=pd.concat([evaluate(p,test,10.,.25,'in_distribution') for p in primary_policies.values()],ignore_index=True)
    pd.testing.assert_frame_equal(primary,repeated,check_exact=True)
    metadata={'version':'1.2.0','source_revision':REVISION,'source_sha256':SOURCE_SHA,'models':MODELS,
              'source_kind':'historical real LLM outputs; offline replay','new_llm_api_calls':0,
              'audit':audit,'bootstrap_replicates':bootstrap,'deterministic_replay_identical':True,
              'python':sys.version,'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-start,
              'limitations':['Frozen 2024 model outputs; no current or fresh inference claims','Historical estimated USD, not current prices or measured latency',
                 'Parsed multiple-choice response selection, not free-text synthesis or human preference validation',
                 'Complete-case filtered sample; upstream grading and training contamination not independently excluded',
                 'Bootstrap conditional on fitted policies; not independent retraining or novel tasks guarantee']}
    protocol=ROOT/'external_protocol.json'
    if protocol.exists():metadata['protocol_sha256']=hashlib.sha256(protocol.read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x))+'\n')
    print(summary(primary).to_string(index=False),flush=True)
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'.cache/routerbench_0shot.pkl')
    p.add_argument('--output',type=Path,default=ROOT/'results/external');p.add_argument('--download',action='store_true');p.add_argument('--bootstrap',type=int,default=2000)
    args=p.parse_args()
    if args.download: download_source(args.source)
    run(args.source,args.output,args.bootstrap)

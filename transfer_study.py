"""v1.5 ProEval replay: six binary tasks plus a quarantined ordinal diagnostic.

Fixed and replaced model pools, not a CERA/Jev reproduction. The four v1.3
datasets keep their published split salt verbatim; gqa and jigsaw are added by
the same rule so the earlier per-dataset results stay comparable.

Only acquired answer equality/INVALID states reach the sequential controller.
Labels and unacquired responses remain inside the evaluator. Archived errors are
supervision for fitting, never live confidence signals. No LLM endpoints called.
"""
from __future__ import annotations
import argparse, ast, hashlib, itertools, json, math, platform, re, sys, time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
N, BUDGET, DEFER = 4, 3, 0.25
MODES = ('single','fixed3','static','myopic','bellman','disagreement','majority3','prompt_top1','cumulative_score')
PRIMARY_DATASETS = ('gsm8k','svamp','mmlu','strategyqa','gqa','jigsaw')
POOL_NAMES = {
    'legacy': ('gpt_4o','claude35_haiku','gemini25_flash','gemma3_12b'),
    'replacement': ('gpt5_2','claude45_sonnet','gemini3_flash','gemma3_27b'),
}
# VQA-style answer normalization for GQA. This is a documented subset of the
# published VQA answer normalizer: punctuation and articles are dropped and
# small number words are folded to digits. It is not the full official
# normalizer (no contraction or period-digit handling), so agreement patterns
# and the strict audit are literal-normalized answers, never semantic matches.
GQA_ARTICLES = frozenset(('a','an','the'))
GQA_NUMBER_WORDS = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
# Jigsaw archival predictions are verbal toxicity verdicts. The archive's own
# binary label column remains the primary error; this only canonicalizes the
# answer identity used for agreement and for the separate strict audit.
JIGSAW_VERDICTS = {'yes':{'yes','true','1','1.0','toxic','y','t'},'no':{'no','false','0','0.0','non-toxic','non toxic','nontoxic','n','f'}}
# The archive's derived verdict is toxic iff the annotator toxic-fraction is
# strictly greater than 0.5. This was verified against the archive's own binary
# label column on every comparable pair, not assumed.
JIGSAW_TOXIC_THRESHOLD = Decimal('0.5')
# The pinned public gqa release contains a handful of cells whose contents are
# a data-generation debug string rather than a model answer. They are treated as
# unparseable answers and counted in the source audit instead of being scored as
# if they were real responses. This is a defect of the upstream artifact, not a
# property of the models.
UPSTREAM_ARTIFACT_MARKERS = ('[DEBUG]',)

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value) -> None:
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.generic): return clean(v.item())
        if isinstance(v,float) and not math.isfinite(v): return None
        return v
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def normalized(text: str) -> str:
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC',str(text)).casefold().split())

def question_text(value: str, dataset: str) -> str:
    if dataset == 'mmlu':
        tree = ast.parse(value, mode='eval').body
        if not isinstance(tree, ast.Dict): raise ValueError('MMLU prompt is not a dict')
        obj = {}
        for key, node in zip(tree.keys, tree.values):
            key = ast.literal_eval(key)
            if key == 'choices' and isinstance(node, ast.Call):
                # The pinned archive stores NumPy array reprs. Decode literal
                # elements only; NEVER eval a constructor or dtype expression.
                if not isinstance(node.func, ast.Name) or node.func.id != 'array' or len(node.args) != 1 or any(k.arg != 'dtype' for k in node.keywords):
                    raise ValueError('Nonliteral choices constructor')
                obj[key] = ast.literal_eval(node.args[0])
                for keyword in node.keywords:
                    if isinstance(keyword.value, ast.Name) and keyword.value.id == 'object': continue
                    if not isinstance(ast.literal_eval(keyword.value), str): raise ValueError('Nonliteral dtype')
            else: obj[key] = ast.literal_eval(node)
        if not isinstance(obj,dict) or 'question' not in obj or 'choices' not in obj:
            raise ValueError('MMLU prompt schema')
        return str(obj['question'])+'\n'+'\n'.join(map(str,obj['choices']))
    if dataset == 'gqa':
        # Literal-evaluated only: the stored prompt is a dict repr with an
        # integer image id. The image id enters the group key so that the same
        # question text asked about different images is not merged.
        tree = ast.parse(value, mode='eval').body
        if not isinstance(tree, ast.Dict): raise ValueError('GQA prompt is not a dict')
        obj = {}
        for key, node in zip(tree.keys, tree.values): obj[ast.literal_eval(key)] = ast.literal_eval(node)
        if 'question' not in obj or 'image_id' not in obj: raise ValueError('GQA prompt schema')
        return str(obj['question'])+'\nimage '+str(obj['image_id'])
    return str(value)

def group_id(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()

def split_id(group: str) -> str:
    n = int.from_bytes(hashlib.sha256(('dtmoa-proeval-v13:'+group).encode()).digest()[:8],'big') / 2**64
    return 'train' if n < .6 else 'dev' if n < .8 else 'test'

def normalize_answer(value, dataset: str) -> str | None:
    """Normalize the archive's answer column only; do not look at gold/reasoning."""
    s = str(value).strip()
    if not s or s.casefold() in ('nan','none','null'): return None
    if any(m in s for m in UPSTREAM_ARTIFACT_MARKERS): return None
    if dataset in ('gsm8k','svamp'):
        s=s.replace(',','')
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?\d+)?',s): return None
        try:
            n=Fraction(s)
            return f'{n.numerator}/{n.denominator}'
        except (ValueError,ZeroDivisionError,OverflowError): return None
    if dataset=='strategyqa':
        return {'yes':'yes','true':'yes','no':'no','false':'no'}.get(s.casefold())
    if dataset=='mmlu':
        return s.upper() if s.upper() in ('A','B','C','D') else None
    if dataset=='gqa':
        import unicodedata
        folded=''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if not unicodedata.combining(c))
        tokens=[GQA_NUMBER_WORDS.get(x,x) for x in re.sub(r'[^0-9a-z]+',' ',folded).split()]
        return ' '.join(x for x in tokens if x not in GQA_ARTICLES) or None
    if dataset=='jigsaw':
        return next((k for k,v in JIGSAW_VERDICTS.items() if s.casefold() in v),None)
    if dataset=='dices':
        try:
            n=Decimal(s)
            return str(int(n)) if n.is_finite() and n==int(n) and 1<=n<=5 else None
        except (ValueError,OverflowError,InvalidOperation): return None
    raise ValueError(dataset)

def normalize_gold(value, dataset: str) -> str | None:
    s=str(value)
    if dataset=='gsm8k':
        return normalize_answer(s.rsplit('####',1)[-1],dataset) if '####' in s else None
    if dataset=='jigsaw':
        # The archive stores the annotator toxic-fraction, not the derived
        # verdict. Threshold it explicitly; the primary error label stays the
        # archive's own binary column.
        try: n=Decimal(s.strip())
        except (ValueError,InvalidOperation,OverflowError): return None
        return None if not n.is_finite() else ('yes' if n>JIGSAW_TOXIC_THRESHOLD else 'no')
    if dataset=='mmlu':
        try:
            d=Decimal(s)
            return 'ABCD'[int(d)] if d==int(d) and 0<=d<=3 else None
        except (ValueError,InvalidOperation,OverflowError): return None
    if dataset=='dices': return None  # ordinal label semantics remain upstream
    return normalize_answer(s,dataset)

@dataclass
class Data:
    dataset: str
    ids: np.ndarray
    groups: np.ndarray
    splits: np.ndarray
    text: np.ndarray
    answers: dict[str,np.ndarray]
    errors: dict[str,np.ndarray]
    strict_errors: dict[str,np.ndarray]
    def subset(self, index):
        return Data(self.dataset,self.ids[index],self.groups[index],self.splits[index],self.text[index],
                    {k:v[index] for k,v in self.answers.items()},
                    {k:v[index] for k,v in self.errors.items()},
                    {k:v[index] for k,v in self.strict_errors.items()})
    def __len__(self): return len(self.ids)

def load_dataset(path: Path, dataset: str, expected_sha: str) -> tuple[Data,dict]:
    if digest(path)!=expected_sha: raise ValueError('Source hash mismatch')
    raw=pd.read_csv(path,keep_default_na=False,dtype=str)
    models=sum((list(v) for v in POOL_NAMES.values()),[])
    required=['index','question','ground_truth']+[v for m in models for v in ('prediction_'+m,'label_'+m)]
    if not set(required).issubset(raw.columns): raise ValueError('Missing input columns')
    d=raw[required].copy()  # no raw reasoning imported into policy features
    if d['index'].duplicated().any(): raise ValueError('Duplicate source IDs')
    scores=d[['label_'+m for m in models]].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    valid=np.isfinite(scores).all(axis=1)&((scores>=0)&(scores<=1)).all(axis=1)
    if dataset!='dices': valid &= np.isin(scores,[0,1]).all(axis=1)
    if not d.question.str.strip().astype(bool).all(): raise ValueError('Empty question')
    d=d.loc[valid].copy();scores=scores[valid]
    text=np.array([question_text(x,dataset) for x in d.question])
    groups=np.array([group_id(x) for x in text]);splits=np.array([split_id(x) for x in groups])
    answers={};errors={};strict={};report={'dataset':dataset,'source_sha256':expected_sha,'source_rows':len(raw),
        'retained_rows':len(d),'excluded_invalid_label':int((~valid).sum()),'unique_groups':len(set(groups)),
        'duplicate_group_rows':len(groups)-len(set(groups)),
        'split_counts':{s:int((splits==s).sum()) for s in ('train','dev','test')},'pools':{}}
    for p,ms in POOL_NAMES.items():
        raw=[[str(v) for v in d['prediction_'+m]] for m in ms]
        a=np.array([[normalize_answer(v,dataset) for v in col] for col in raw],dtype=object).T
        e=d[['label_'+m for m in ms]].astype(float).to_numpy()
        g=np.array([normalize_gold(v,dataset) for v in d.ground_truth],dtype=object)
        se=np.where(g[:,None]!=None,(a!=g[:,None]).astype(float),np.nan) if dataset!='dices' else np.full(e.shape,np.nan)
        answers[p]=a;errors[p]=e;strict[p]=se
        report['pools'][p]={'models':ms,'invalid_answers_per_model':dict(zip(ms,(a==None).sum(axis=0).tolist())),
            'upstream_artifact_answer_cells':int(sum(any(m in cell for m in UPSTREAM_ARTIFACT_MARKERS) for col in raw for cell in col)),
            'strict_gold_unparseable_rows':int(sum(x is None for x in g)) if dataset!='dices' else None,
            'strict_vs_upstream_error_disagreements':int(np.sum(np.isfinite(se)&(se!=e))) if dataset!='dices' else None}
    for s in ('train','dev','test'):
        if (splits==s).sum()<10: raise ValueError('Insufficient split support')
    return Data(dataset,d['index'].to_numpy(),groups,splits,text,answers,errors,strict),report

def canonical(values) -> tuple[int,...]:
    groups={};answer=[]
    for value in values:
        if value is None: answer.append(-1)
        else:
            if value not in groups: groups[value]=len(groups)
            answer.append(groups[value])
    return tuple(answer)

def full_patterns(n=N):
    result=[]
    def visit(prefix,max_value):
        if len(prefix)==n: result.append(tuple(prefix));return
        for value in (-1,*range(max_value+2)):
            visit(prefix+[value],max(max_value,value))
    visit([],-1)
    return tuple(result)

PATTERNS=full_patterns()
PATTERN_ID={p:i for i,p in enumerate(PATTERNS)}
ROOT_STATE=(0,())

def project(pattern,mask):
    values=[None if pattern[i]<0 else pattern[i] for i in range(N) if mask>>i&1]
    return mask,canonical(values)

STATES=tuple(sorted({project(p,m) for p in PATTERNS for m in range(1<<N)},key=lambda s:(s[0].bit_count(),s)))
STATE_ID={s:i for i,s in enumerate(STATES)}
STATE_MASK=np.array([s[0] for s in STATES])
MATCH=np.array([[project(p,s[0])==s for p in PATTERNS] for s in STATES],dtype=float)
LEGAL={};CHILDREN={}
for s in STATES:
    left=[i for i in range(N) if not s[0]>>i&1]
    LEGAL[s]=tuple(batch for k in range(1,min(BUDGET-s[0].bit_count(),len(left))+1) for batch in itertools.combinations(left,k))
    for batch in LEGAL[s]:
        mask=s[0]|sum(1<<i for i in batch)
        CHILDREN[s,batch]=tuple(sorted({project(p,mask) for p in PATTERNS if project(p,s[0])==s}))

class World:
    def __init__(self, answers, errors, strength=10.):
        a=np.asarray(answers,dtype=object);e=np.asarray(errors,dtype=float)
        if a.ndim!=2 or a.shape[1]!=N or a.shape!=e.shape or not len(a): raise ValueError('Expected aligned nonempty n by four arrays')
        if not np.isfinite(e).all() or np.any((e<0)|(e>1)) or not np.isfinite(strength) or strength<=0: raise ValueError('Invalid error/prior')
        bins=np.array([PATTERN_ID[canonical(row)] for row in a])
        counts=np.bincount(bins,minlength=len(PATTERNS)).astype(float)
        self.global_error=(e.sum(axis=0)+.5)/(len(e)+1)
        # Uniform positive support prior over canonical patterns, not semantic truth.
        mass=counts+strength/len(PATTERNS)
        sums=np.column_stack([np.bincount(bins,weights=e[:,i],minlength=len(PATTERNS)) for i in range(N)])
        sums += (strength/len(PATTERNS))*self.global_error
        self.mass=MATCH@mass
        self.risk=(MATCH@sums)/self.mass[:,None]
    def terminal(self,state,defer=DEFER):
        seen=[i for i in range(N) if state[0]>>i&1]
        options=[(float(self.risk[STATE_ID[state],i]),i) for i in seen]+[(defer,-1)]
        return min(options,key=lambda x:(x[0],x[1]))
    def transitions(self,state,batch):
        ch=CHILDREN[state,batch]
        masses=np.array([self.mass[STATE_ID[s]] for s in ch])
        return tuple(zip(ch,masses/masses.sum()))

class Policy:
    def __init__(self,world:World,mode:str,price=.01,defer=DEFER,threshold=1.4):
        if mode not in MODES: raise ValueError('Unknown mode')
        if not np.isfinite([price,defer]).all() or price<0 or not 0<=defer<=1: raise ValueError('Invalid objective')
        self.world,self.mode,self.price,self.defer,self.threshold=world,mode,float(price),float(defer),threshold
        self.actions={};self.values={}
        if mode in ('bellman','myopic'):
            for state in reversed(STATES):
                risk,_=world.terminal(state,defer);best=(risk,())
                for batch in LEGAL[state]:
                    if mode=='myopic' and len(batch)!=1: continue
                    q=price*len(batch)+sum(p*(self.values[ch] if mode=='bellman' else world.terminal(ch,defer)[0]) for ch,p in world.transitions(state,batch))
                    if q<best[0]-1e-12: best=q,batch
                self.values[state],self.actions[state]=best
        if mode in ('single','fixed3','static'):
            sizes={'single':(1,),'fixed3':(3,),'static':(0,1,2,3)}[mode]
            best=(defer,()) if 0 in sizes else (float('inf'),())
            for batch in LEGAL[ROOT_STATE]:
                if len(batch) not in sizes: continue
                q=price*len(batch)+sum(p*world.terminal(ch,defer)[0] for ch,p in world.transitions(ROOT_STATE,batch))
                if q<best[0]-1e-12:best=q,batch
            self.actions[ROOT_STATE]=best[1];self.values[ROOT_STATE]=best[0]
        self.rank=tuple(np.argsort(world.global_error,kind='stable').tolist())
    def next_batch(self,state,prompt_risks=None):
        if self.mode in ('bellman','myopic','single','fixed3','static'): return self.actions.get(state,())
        if self.mode=='disagreement':
            if state==ROOT_STATE:return self.rank[:2]
            vals=state[1]
            if state[0].bit_count()==2 and (-1 in vals or len(set(vals))>1):return (self.rank[2],)
            return ()
        if self.mode=='majority3':return self.rank[:3] if state==ROOT_STATE else ()
        if self.mode in ('prompt_top1','cumulative_score'):
            if state!=ROOT_STATE:return ()
            if prompt_risks is None or len(prompt_risks)!=N or not np.isfinite(prompt_risks).all():raise ValueError('Prompt estimate missing')
            ix=tuple(np.argsort(prompt_risks,kind='stable').tolist())
            if self.mode=='prompt_top1':return (ix[0],) if prompt_risks[ix[0]]+self.price<self.defer else ()
            total=0.;batch=[]
            for i in ix[:BUDGET]:
                batch.append(i);total+=1-float(prompt_risks[i])
                if total>=self.threshold:break
            return tuple(batch)
        raise AssertionError(self.mode)
    def choose(self,state,observed):
        risk,i=self.world.terminal(state,self.defer)
        if self.mode=='majority3':
            valid=[i for i in observed if observed[i] is not None]
            if not valid:return self.defer,-1
            counts=Counter(observed[i] for i in valid)
            i=min(valid,key=lambda i:(-counts[observed[i]],self.world.global_error[i],i))
            risk=float(self.world.risk[STATE_ID[state],i])
        return risk,i
    def execute(self,acquire:Callable[[int],str|None],prompt_risks=None):
        seen={};batches=[];state=ROOT_STATE
        for _ in range(BUDGET+1):
            batch=self.next_batch(state,prompt_risks)
            if not batch:
                risk,choice=self.choose(state,seen)
                if choice>=0 and choice not in seen:raise AssertionError('Unacquired output')
                return choice,risk,tuple(seen),tuple(batches),state
            if len(seen)+len(batch)>BUDGET or any(i in seen or i not in range(N) for i in batch) or len(set(batch))!=len(batch):raise AssertionError('Illegal query')
            for i in batch:seen[i]=acquire(i)
            batches.append(tuple(batch))
            state=(sum(1<<i for i in seen),canonical(seen[i] for i in sorted(seen)))
        raise AssertionError('Did not stop')

class PromptHead:
    """Transparent CPU baseline; neither CERA hidden states nor the Jev model."""
    def fit(self,text,errors):
        self.hash=HashingVectorizer(n_features=2048,analyzer='char',ngram_range=(3,5),alternate_sign=False,norm=None)
        self.tfidf=TfidfTransformer();x=self.tfidf.fit_transform(self.hash.transform(text))
        self.reg=Ridge(alpha=10.,solver='lsqr',tol=1e-8).fit(x,errors)
        return self
    def predict(self,text):
        return np.clip(self.reg.predict(self.tfidf.transform(self.hash.transform(text))),0.,1.)

def evaluate(policy,data,pool,phase,prompt_risks=None):
    rows=[];a=data.answers[pool];e=data.errors[pool];gold=data.strict_errors[pool]
    for r in range(len(data)):
        choice,risk,queried,batches,state=policy.execute(lambda i:a[r,i],None if prompt_risks is None else prompt_risks[r])
        loss=policy.defer if choice<0 else float(e[r,choice])
        strict=policy.defer if choice<0 else float(gold[r,choice])
        rows.append({'dataset':data.dataset,'sample_id':str(data.ids[r]),'group':data.groups[r],'phase':phase,'policy':policy.mode,
            'price':policy.price,'defer_loss':policy.defer,'selected':choice,'answered':int(choice>=0),'loss':loss,
            'strict_loss':strict,'predicted_error':risk if choice>=0 else np.nan,'queries':len(queried),
            'cost':policy.price*len(queried),'objective':loss+policy.price*len(queried),
            'strict_objective':strict+policy.price*len(queried),'query_order':','.join(map(str,queried)),
            'batches':';'.join(','.join(map(str,b)) for b in batches),'state':str(state)})
    return pd.DataFrame(rows)

def calibrate(frame):
    g=frame[frame.answered==1]
    if not len(g):return np.nan,np.nan
    p=1-g.predicted_error.to_numpy();y=1-g.loss.to_numpy()
    if not np.isin(y,[0,1]).all():raise ValueError('Brier correctness requires binary truth')
    brier=float(np.mean((p-y)**2));bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.mean(bins==i)*abs(float(p[bins==i].mean()-y[bins==i].mean())) for i in range(10) if np.any(bins==i))
    return brier,ece

def summarize(frame):
    rows=[]
    for keys,g in frame.groupby(['dataset','phase','policy','price'],sort=True):
        d,phase,mode,price=keys
        answered=g[g.answered==1]
        brier,ece=calibrate(g) if d!='dices' else (np.nan,np.nan)
        rows.append(dict(dataset=d,phase=phase,policy=mode,price=price,n=len(g),objective=g.objective.mean(),loss=g.loss.mean(),
            coverage=g.answered.mean(),selected_mean_error=answered.loss.mean(),queries=g.queries.mean(),cost=g.cost.mean(),
            strict_objective=g.strict_objective.mean() if d!='dices' else np.nan,brier=brier,ece=ece))
    return pd.DataFrame(rows)

def select(train,dev,pool,price,modes=MODES):
    worlds={s:World(train.answers[pool],train.errors[pool],s) for s in (1.,10.,100.)}
    head=PromptHead().fit(train.text,train.errors[pool]) if any(m.startswith('prompt_') or m=='cumulative_score' for m in modes) else None
    risks=head.predict(dev.text) if head else None
    policies={};records=[]
    for mode in modes:
        candidates=[]
        for strength,w in worlds.items():
            for threshold in ((.7,1.4,2.1) if mode=='cumulative_score' else (1.4,)):
                p=Policy(w,mode,price,threshold=threshold)
                val=float(evaluate(p,dev,pool,'dev',risks).objective.mean())
                candidates.append((val,strength,threshold,p))
        value,strength,threshold,p=min(candidates,key=lambda r:r[:3])
        policies[mode]=p
        records.append(dict(dataset=train.dataset,pool=pool,price=price,policy=mode,strength=strength,threshold=threshold,dev_objective=value))
    return policies,head,records

def comparison(frame,count=2000):
    contrasts=[('replacement_refit','bellman','replacement_refit','static'),('replacement_refit','bellman','replacement_refit','myopic'),
               ('replacement_frozen','bellman','replacement_refit','bellman')]
    result=[]
    for pa,a,pb,b in contrasts:
        draws=[];diffs=[]
        for dataset_index,dataset in enumerate(PRIMARY_DATASETS):
            x=frame[(frame.dataset==dataset)&(frame.phase==pa)&(frame.policy==a)].set_index('sample_id')
            y=frame[(frame.dataset==dataset)&(frame.phase==pb)&(frame.policy==b)].set_index('sample_id').reindex(x.index)
            if len(x)<2 or y.objective.isna().any():raise ValueError('Unpaired comparison')
            diff=x.objective-y.objective
            grouped=pd.DataFrame({'group':x.group,'diff':diff}).groupby('group')['diff'].agg(['sum','count'])
            rng=np.random.default_rng(np.random.SeedSequence([130918,dataset_index]));boot=[];n=len(grouped)
            sums=grouped['sum'].to_numpy();counts=grouped['count'].to_numpy()
            for start in range(0,count,100):
                ix=rng.integers(0,n,(min(100,count-start),n));boot.extend(sums[ix].sum(axis=1)/counts[ix].sum(axis=1))
            lo,hi=np.quantile(boot,[.025,.975]);mean=float(diff.mean())
            result.append(dict(dataset=dataset,a=pa+':'+a,b=pb+':'+b,difference=mean,lo=lo,hi=hi,n_groups=n,bootstrap=count))
            draws.append(boot);diffs.append(mean)
        lo,hi=np.quantile(np.mean(draws,axis=0),[.025,.975])
        result.append(dict(dataset='macro_binary',a=pa+':'+a,b=pb+':'+b,difference=np.mean(diffs),lo=lo,hi=hi,n_groups=np.nan,bootstrap=count))
    return pd.DataFrame(result)

def run(cache:Path,out:Path,bootstrap=2000):
    protocol=json.loads((ROOT/'transfer_protocol.json').read_text());out.mkdir(parents=True,exist_ok=True)
    all_data=[];audits=[];frames=[];selections=[];sweeps=[];refresh=[];cpu=[];fit_sizes=[]
    for dataset in protocol['datasets']:
        data,audit=load_dataset(cache/f'{dataset}_predictions.csv',dataset,protocol['source_sha256'][dataset]);all_data.append(data);audits.append(audit)
        train=data.subset(data.splits=='train');dev=data.subset(data.splits=='dev');test=data.subset(data.splits=='test')
        pd.DataFrame({'sample_id':data.ids,'group':data.groups,'split':data.splits}).to_csv(out/f'split_{dataset}.csv',index=False)
        legacy=None;legacy_head=None;legacy_selection=None;frozen=None
        for pool,phase in (('legacy','legacy_id'),('replacement','replacement_refit')):
            start=time.perf_counter();policies,head,recs=select(train,dev,pool,.01);train_seconds=time.perf_counter()-start
            risks=head.predict(test.text)
            selections.extend(recs)
            if pool=='legacy':legacy,legacy_head,legacy_selection=policies,head,recs
            for mode,p in policies.items():
                start=time.perf_counter();f=evaluate(p,test,pool,phase,risks);cpu.append(dict(dataset=dataset,phase=phase,policy=mode,seconds=time.perf_counter()-start,n=len(test),scope='Python replay+ledger overhead, not LLM service latency'))
                frames.append(f)
            fit_sizes.append(dict(dataset=dataset,pool=pool,train_questions=len(train),dev_questions=len(dev),archive_answers_acquired=4*(len(train)+len(dev)),assumed_calibration_cost=.01*4*(len(train)+len(dev)),train_select_seconds=train_seconds))
            if pool=='replacement':
                for price in (0.,.005,.02,.05):
                    ps,h,_=select(train,dev,pool,price,modes=('single','static','myopic','bellman'))
                    sweeps.append(summarize(pd.concat([evaluate(p,test,pool,'replacement_refit',None) for p in ps.values()],ignore_index=True)))
        frozen_risks=legacy_head.predict(test.text)
        for p in legacy.values():frames.append(evaluate(p,test,'replacement','replacement_frozen',frozen_risks))
        # Fixed non-test sample budgets; no target dev/test labels used for refresh.
        order=np.argsort(train.groups,kind='stable')
        for budget in (50,200):
            ix=order[:min(budget,len(order))];cal=train.subset(ix)
            for mode in ('single','static','myopic','bellman'):
                rec=next(r for r in legacy_selection if r['policy']==mode)
                p=Policy(World(cal.answers['replacement'],cal.errors['replacement'],rec['strength']),mode,.01)
                f=evaluate(p,test,'replacement',f'refresh_{budget}')
                frames.append(f)
                base=next(f for f in reversed(frames) if f.phase.iloc[0]=='replacement_frozen' and f.policy.iloc[0]==mode)
                gain=float(base.objective.mean()-f.objective.mean());charge=4*len(cal)*.01
                refresh.append(dict(dataset=dataset,policy=mode,budget=budget,actual_calibration_questions=len(cal),new_archive_answers=4*len(cal),assumed_refresh_cost=charge,
                                    per_query_gain=gain,break_even_future_queries=math.ceil(charge/gain) if gain>0 else np.nan,
                                    amortized_objective_1000=float(f.objective.mean()+charge/1000)))
        print('completed',dataset,flush=True)
    manifest=pd.concat([pd.DataFrame({'dataset':d.dataset,'id':d.ids,'group':d.groups,'split':d.splits}) for d in all_data],ignore_index=True)
    if (manifest.groupby('group')['split'].nunique()>1).any():raise AssertionError('Cross-dataset split leakage')
    manifest.to_csv(out/'split_manifest.csv',index=False)
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(out/'per_case.csv',index=False);s=summarize(frame);s.to_csv(out/'summary.csv',index=False)
    # Joint duplicate groups across binary datasets would require a shared bootstrap weight.
    # Fail rather than silently treating such clusters as independent.
    binary_manifest=manifest[manifest.dataset.isin(PRIMARY_DATASETS)]
    cross=int((binary_manifest.groupby('group').dataset.nunique()>1).sum())
    if cross:raise AssertionError('Cross-dataset repeated groups require joint bootstrap')
    # Per-dataset discrimination diagnostic. A task where every decision-theoretic
    # policy takes the same action carries no comparison information; that is a
    # property of the archived answers and the declared objective, and it is
    # reported rather than silently dropped from the primary average.
    core=frame[frame.policy.isin(['single','static','myopic','bellman'])]
    diag=core.groupby(['dataset','phase','price'],as_index=False).agg(objective_min=('objective','min'),objective_max=('objective','max'),mean_queries=('queries','mean'),deferral_rate=('answered',lambda s:1.-s.mean()),n=('objective','size'))
    diag['objective_spread']=diag.objective_max-diag.objective_min
    diag['policies_agree_exactly']=diag.objective_spread<=1e-12
    diag.to_csv(out/'policy_discrimination.csv',index=False)
    comparison(frame,bootstrap).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'selection.csv',index=False)
    pd.concat(sweeps,ignore_index=True).to_csv(out/'cost_sweep.csv',index=False)
    pd.DataFrame(refresh).to_csv(out/'refresh_costs.csv',index=False)
    pd.DataFrame(cpu).to_csv(out/'controller_timing.csv',index=False)
    pd.DataFrame(fit_sizes).to_csv(out/'calibration_resources.csv',index=False)
    write_json(out/'source_audit.json',audits)
    # Numeric/categorical derivatives; do not redistribute original raw responses.
    for d in all_data:
        for pool in POOL_NAMES:
            np.savez_compressed(out/f'traces_{d.dataset}_{pool}.npz',
                patterns=np.array([canonical(row) for row in d.answers[pool]],dtype=np.int8),
                errors=d.errors[pool],strict_errors=d.strict_errors[pool],ids=d.ids.astype(str),groups=d.groups.astype(str),splits=d.splits.astype(str))
    write_json(out/'provenance.json',dict(version='1.5.0',protocol_sha256=digest(ROOT/'transfer_protocol.json'),
        source_revision=protocol['source_revision'],new_llm_api_calls=0,jev_calls=0,cera_agent_training_steps=0,
        binary_questions=sum(len(d) for d in all_data if d.dataset in PRIMARY_DATASETS),ordinal_questions=sum(len(d) for d in all_data if d.dataset=='dices'),
        archived_response_records=sum(len(d)*8 for d in all_data),binary_test_questions=int(((manifest.split=='test')&manifest.dataset.isin(PRIMARY_DATASETS)).sum()),
        cross_split_group_overlap=0,cross_binary_dataset_groups=cross,python=sys.version,platform=platform.platform(),
        limitations=protocol['limits']))
    print(s[(s.phase=='replacement_refit')&(s.policy.isin(['single','static','myopic','bellman']))].to_string(index=False),flush=True)

def download(cache:Path):
    p=json.loads((ROOT/'transfer_protocol.json').read_text());cache.mkdir(parents=True,exist_ok=True)
    for name in p['datasets']:
        path=cache/f'{name}_predictions.csv'
        if not path.exists():
            url=f"https://raw.githubusercontent.com/google-deepmind/proeval/{p['source_revision']}/data/{name}_predictions.csv"
            req=urllib.request.Request(url,headers={'User-Agent':'DTMoA research replay'})
            with urllib.request.urlopen(req,timeout=180) as r:content=r.read()
            tmp=path.with_suffix('.partial');tmp.write_bytes(content)
            if digest(tmp)!=p['source_sha256'][name]:tmp.unlink();raise ValueError('Downloaded source changed')
            tmp.replace(path)
        if digest(path)!=p['source_sha256'][name]:raise ValueError('Cached source changed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=ROOT/'.cache/proeval');parser.add_argument('--out',type=Path,default=ROOT/'results/transfer');parser.add_argument('--download',action='store_true');parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.download:download(args.cache)
    run(args.cache,args.out,args.bootstrap)

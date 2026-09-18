"""Direction/confidence-aware finite acquisition and compiled execution.
Conditional finite-model optimality is not real-world architectural optimality.
"""
from __future__ import annotations
import itertools,json
from dataclasses import dataclass
import numpy as np
M=3

def observations(probabilities,bins=2,threshold=.75):
    p=np.asarray(probabilities,float)
    if p.ndim!=3 or p.shape[1]!=M or p.shape[2]<2 or bins not in (1,2):raise ValueError('Expected [cases,3,options] and one/two bins')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=1e-6):raise ValueError('Invalid probability distributions')
    return p.argmax(-1)*bins+((p.max(-1)>=threshold).astype(int) if bins==2 else 0)

@dataclass
class Compiled:
    k:int
    bins:int
    action:np.ndarray
    choice:np.ndarray
    terminal:np.ndarray
    value:np.ndarray
    mass:np.ndarray
    costs:np.ndarray
    defer:float
    def index(self,state):
        a=self.k*self.bins+1
        if len(state)!=M or any(type(x) not in (int,np.int64,np.int32) or not 0<=x<a for x in state):raise ValueError('Illegal state')
        return int(state[0]*a*a+state[1]*a+state[2])
    def execute(self,acquire,recompute=False):
        state=[0]*M;used=[];a=self.k*self.bins+1
        for _ in range(M+1):
            idx=self.index(state);n=int(self.action[idx])
            if recompute:
                best=float(self.terminal[idx]);n=-1
                for j in range(M):
                    if state[j]:continue
                    indices=[]
                    for o in range(a-1):
                        s=state.copy();s[j]=o+1;indices.append(self.index(s))
                    weights=self.mass[indices]/self.mass[idx];q=float(self.costs[j]+weights@self.value[indices])
                    if q<best-1e-12:best=q;n=j
            if n<0:
                selected=int(self.choice[idx])
                if selected>=0 and selected not in [(state[i]-1)//self.bins for i in used]:raise AssertionError('Unacquired candidate emitted')
                return selected,tuple(used),float(self.terminal[idx])
            if n in used or n not in range(M):raise AssertionError('Illegal acquisition')
            obs=int(acquire(n))
            if not 0<=obs<self.k*self.bins:raise ValueError('Invalid observation')
            state[n]=obs+1;used.append(n)
        raise AssertionError('Did not stop')
    def save(self,path,contract):
        meta={'k':self.k,'bins':self.bins,'defer':self.defer,'contract':contract}
        np.savez_compressed(path,action=self.action,choice=self.choice,terminal=self.terminal,value=self.value,mass=self.mass,costs=self.costs,metadata=np.array(json.dumps(meta,sort_keys=True)))
    @classmethod
    def load(cls,path,contract):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            if meta['contract']!=json.loads(json.dumps(contract,sort_keys=True)):raise ValueError('Model/data version contract mismatch')
            arrays={k:z[k].copy() for k in ('action','choice','terminal','value','mass','costs')}
        expected=(meta['k']*meta['bins']+1)**M
        if any(len(arrays[k])!=expected for k in ('action','choice','terminal','value','mass')):raise ValueError('Invalid table dimensions')
        if not all(np.isfinite(v).all() for v in arrays.values()) or (arrays['mass']<=0).any():raise ValueError('Invalid compiled values')
        if not np.isin(arrays['action'],[-1,0,1,2]).all() or not np.isin(arrays['choice'],range(-1,meta['k'])).all():raise ValueError('Invalid actions')
        return cls(meta['k'],meta['bins'],**arrays,defer=meta['defer'])

class JointWorld:
    def __init__(self,probs,y,bins=2,strength=32.,missing=False):
        p=np.asarray(probs,float);y=np.asarray(y,int)
        obs=observations(p,bins);self.k=p.shape[2];self.bins=bins;self.a=self.k*bins;self.yk=self.k+int(missing)
        if len(y)!=len(p) or not len(y) or (y<0).any() or (y>=self.yk).any() or strength<=0 or np.isnan(strength):raise ValueError('Bad calibration data')
        n=len(p);k=self.k;a=self.a;py=np.bincount(y,minlength=self.yk)+.5;py=py/py.sum();likelihood=[]
        for m in range(M):
            correct=(obs[:,m]//bins==y);acc=(correct.sum()+1)/(n+2)
            conf=np.bincount(obs[:,m]%bins,minlength=bins)+1;conf=conf/conf.sum();prior=np.zeros((self.yk,a))
            for truth in range(self.yk):
                for pred in range(k):
                    q=1/k if truth==k else (acc if pred==truth else (1-acc)/(k-1))
                    prior[truth,pred*bins:(pred+1)*bins]=q*conf
            counts=prior*4;np.add.at(counts,(y,obs[:,m]),1.);likelihood.append(counts/counts.sum(1,keepdims=True))
        joint=py[:,None,None,None]*likelihood[0][:,:,None,None]*likelihood[1][:,None,:,None]*likelihood[2][:,None,None,:]
        if np.isfinite(strength):
            joint*=strength;np.add.at(joint,(y,obs[:,0],obs[:,1],obs[:,2]),1.);joint/=joint.sum()
        self.joint=joint
    def compile(self,costs=(.01,.01,.01),defer=.25,mode='bellman'):
        costs=np.asarray(costs,float)
        if costs.shape!=(M,) or not np.isfinite(costs).all() or (costs<0).any() or not 0<=defer<=1 or mode not in ('bellman','myopic'):raise ValueError('Bad objective')
        a=self.a;base=a+1;length=base**M
        action=np.full(length,-1,dtype=np.int8);choice=np.full(length,-1,dtype=np.int16);risk=np.zeros(length);value=np.zeros(length);mass=np.zeros(length)
        marginal={mask:self.joint.sum(axis=tuple(i+1 for i in range(M) if not mask>>i&1)) for mask in range(1<<M)}
        def index(s):return s[0]*base*base+s[1]*base+s[2]
        for mask in sorted(range(1<<M),key=int.bit_count,reverse=True):
            seen=[i for i in range(M) if mask>>i&1]
            for values in itertools.product(range(a),repeat=len(seen)):
                state=[0]*M
                for i,v in zip(seen,values):state[i]=v+1
                idx=index(state);weights=marginal[mask][(slice(None),*values)];den=float(weights.sum());mass[idx]=den
                options=[(defer,-1)]+[(1-float(weights[j]/den),j) for j in sorted({v//self.bins for v in values})]
                best,emit=min(options);choice[idx]=emit;risk[idx]=best
                for j in range(M):
                    if mask>>j&1:continue
                    children=[]
                    for o in range(a):
                        next_state=state.copy();next_state[j]=o+1;children.append(index(next_state))
                    q=float(costs[j]+(mass[children]/den)@(value[children] if mode=='bellman' else risk[children]))
                    if q<best-1e-12:best=q;action[idx]=j
                value[idx]=best
        return Compiled(self.k,self.bins,action,choice,risk,value,mass,costs,defer)

class AgreementWorld:
    """v1.3-style equality-only ablation; not the earlier task results."""
    def __init__(self,probs,y,strength=10.):
        p=np.asarray(probs,float);observations(p);self.pred=p.argmax(-1);self.err=(self.pred!=np.asarray(y)[:,None]).astype(float)
        self.global_error=(self.err.sum(0)+.5)/(len(p)+1)
        def canon(vals):
            mapping={};return tuple(mapping.setdefault(int(v),len(mapping)) for v in vals)
        self.canon=canon;patterns=sorted(set(canon(v) for v in itertools.product(range(3),repeat=3)));self.patterns=patterns
        counts=np.full(len(patterns),strength/len(patterns));sums=np.tile(self.global_error,(len(patterns),1))*counts[:,None]
        for pred,error in zip(self.pred,self.err):
            j=patterns.index(canon(pred));counts[j]+=1;sums[j]+=error
        self.counts=counts;self.sums=sums
    def execute(self,acquire,cost=.01,defer=.25):
        from functools import lru_cache
        @lru_cache(None)
        def stats(seen,values):
            ok=[j for j,p in enumerate(self.patterns) if self.canon(p[i] for i in seen)==values]
            den=float(self.counts[ok].sum());errors=self.sums[ok].sum(0)/den;return ok,den,errors
        @lru_cache(None)
        def value(seen,values):
            ok,den,errors=stats(seen,values);risk,selected=min([(defer,-1)]+[(float(errors[i]),i) for i in seen]);best=risk;next_model=-1
            for i in range(M):
                if i in seen:continue
                new=tuple(sorted((*seen,i)));parts={}
                for j in ok:
                    v=self.canon(self.patterns[j][x] for x in new);parts[v]=parts.get(v,0.)+self.counts[j]
                q=cost+sum(w/den*value(new,v)[0] for v,w in parts.items())
                if q<best-1e-12:best=q;next_model=i
            return best,next_model,selected,risk
        seen={};used=[]
        for _ in range(M+1):
            keys=tuple(sorted(seen));vals=self.canon(seen[i] for i in keys);_,n,sel,risk=value(keys,vals)
            if n<0:return (-1 if sel<0 else seen[sel]),tuple(used),risk
            seen[n]=int(acquire(n));used.append(n)
        raise AssertionError('Budget')

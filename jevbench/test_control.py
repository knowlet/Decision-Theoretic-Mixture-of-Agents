import itertools,json
from functools import lru_cache
import numpy as np
import pytest
from jevbench.control import JointWorld,Compiled,observations,AgreementWorld
from jevbench.fixtures import choose,digest,group
from jevbench.run import validate_score

def sample(k=3,n=24,seed=10):
    rng=np.random.default_rng(seed);p=rng.dirichlet(np.ones(k),(n,3));y=rng.integers(0,k,n);return p,y

@pytest.mark.parametrize('k',[2,3,4,9])
@pytest.mark.parametrize('bins',[1,2])
def test_lookup_matches_online_q(k,bins):
    p,y=sample(k);w=JointWorld(p,y,bins=bins);c=w.compile(defer=.75);obs=observations(p,bins)
    for row in obs:assert c.execute(lambda i:int(row[i]))==c.execute(lambda i:int(row[i]),recompute=True)

@pytest.mark.parametrize('seed',range(3))
def test_independent_enumerated_solver(seed):
    p,y=sample(2,seed=seed);w=JointWorld(p,y,bins=1);joint=w.joint;cost=.01;defer=.75
    @lru_cache(None)
    def f(state):
        index=tuple(slice(None) if x<0 else x for x in state);posterior=joint[(slice(None),*index)];mass=float(posterior.sum());weights=posterior.sum(axis=tuple(range(1,posterior.ndim)))
        best=min([defer]+[1-float(weights[x]/mass) for x in set(state) if x>=0])
        for i,x in enumerate(state):
            if x>=0:continue
            value=cost
            for obs in range(2):
                child=list(state);child[i]=obs;ci=tuple(slice(None) if z<0 else z for z in child);prob=float(joint[(slice(None),*ci)].sum())/mass;value+=prob*f(tuple(child))
            best=min(best,value)
        return best
    assert w.compile(defer=defer).value[0]==pytest.approx(f((-1,-1,-1)),abs=1e-12)

def test_only_callback_observations_visible():
    p,y=sample();c=JointWorld(p,y).compile(defer=.9);used=[]
    def observe(i):
        assert i not in used;used.append(i);return 1
    label,queries,risk=c.execute(observe)
    assert tuple(used)==queries and len(used)<=3 and (label==-1 or label==0)

def test_high_cost_stops_without_query():
    p,y=sample();c=JointWorld(p,y).compile(costs=(2,2,2));assert c.execute(lambda _:pytest.fail('unexpected call'))[0]==-1

def test_model_version_contract(tmp_path):
    p,y=sample();c=JointWorld(p,y).compile();path=tmp_path/'p.npz';contract={'models':('a','b','c'),'fixture':'f','loss':.25};c.save(path,contract)
    again=Compiled.load(path,contract);np.testing.assert_array_equal(c.action,again.action)
    with pytest.raises(ValueError):Compiled.load(path,contract|{'fixture':'other'})

@pytest.mark.parametrize('invalid',[-.1,float('nan'),2.])
def test_bad_probability_rejected(invalid):
    p,y=sample();p[0,0,0]=invalid
    with pytest.raises(ValueError):observations(p)

def test_absolute_direction_not_just_equality():
    p=np.array([[[.9,.1]]*3]*20+[[[.1,.9]]*3]*5);y=np.zeros(25,dtype=int)
    blind=AgreementWorld(p,y);joint=JointWorld(p,y,bins=1).compile(defer=.25)
    zero=joint.execute(lambda _:0);one=joint.execute(lambda _:1)
    assert zero[0]==0 and one[0]==-1
    assert blind.execute(lambda _:0)[0]==0 and blind.execute(lambda _:1)[0]==1

def test_shortlist_missing_gold_is_not_inserted():
    p,y=sample(9);y[:3]=9;w=JointWorld(p,y,missing=True);assert w.yk==10
    c=w.compile(defer=1.);obs=observations(p)
    for row in obs:assert c.execute(lambda i:int(row[i]))[0]!=9

def test_joint_normalized_positive():
    p,y=sample();w=JointWorld(p,y);assert np.isclose(w.joint.sum(),1) and (w.joint>0).all()

def test_bellman_no_worse_than_myopic_inside_same_model():
    p,y=sample();w=JointWorld(p,y);assert w.compile(defer=.9).value[0]<=w.compile(defer=.9,mode='myopic').value[0]+1e-12

def test_group_sampling_never_reads_gold():
    rows=[{'group':group(str(i)),'source_id':str(i),'gold':i%3} for i in range(50)];first=[r['group'] for r in choose(rows,10)]
    for r in rows:r['gold']=42
    assert first==[r['group'] for r in choose(rows,10)]

def test_group_isolation():
    rows=[{'group':group(str(i//2)),'source_id':str(i)} for i in range(30)];fit=choose(rows,5);dev=choose(rows,5,[r['group'] for r in fit]);assert not {r['group'] for r in fit}&{r['group'] for r in dev}

def test_output_contract():
    req={'id':'x','options':[{'id':'a'},{'id':'b'}]};score={'id':'x','option_ids':['a','b'],'probabilities':[.1,.9],'total_ms':2}
    assert validate_score(score,req)=='b'
    with pytest.raises(ValueError):validate_score(score|{'option_ids':['a','unknown']},req)

def test_confidence_bins_direction():
    p=np.array([[[.8,.2],[.3,.7],[.1,.9]]]);assert observations(p).tolist()==[[1,2,3]]

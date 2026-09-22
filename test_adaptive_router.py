import itertools
import copy
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import adaptive_router as a
import transfer_study as t
import optimizer_study as s

POOL=a.Pool(('a','b','c','d'),'version-1')

def data(seed=16,n=160):
    rng=np.random.default_rng(seed)
    responses=rng.choice(['yes','no',None],(n,4)).astype(object)
    gold=rng.choice(['yes','no'],n)
    return responses,(responses!=gold[:,None]).astype(float)

@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('mode',['bellman','myopic','static','single','fixed3'])
def test_exact_compilation(seed,mode):
    responses,errors=data(seed)
    old=t.Policy(t.World(responses,errors),'bellman' if mode=='bellman' else mode)
    new=a.CompiledPolicy.from_legacy(old,POOL)
    assert s.check_compilation(old,new,POOL)==52
    restored=a.CompiledPolicy.from_json(new.to_json(),expected_pool=POOL)
    assert s.check_compilation(old,restored,POOL)==52

@pytest.mark.parametrize('kind',['equality','polarity'])
def test_probabilities_coherent(kind):
    responses,errors=data();joint=a.fit_joint(responses,errors,kind)
    p,states,idx,match,actions,children=a.graph(kind)
    masses=match@joint.mass
    for state in states:
        for batch in actions[state]:
            assert sum(masses[idx[ch]] for ch in children[state,batch])==pytest.approx(masses[idx[state]])


def test_polarity_retains_predictive_information():
    answers=np.array([['yes']*4]*100+[['no']*4]*100,dtype=object)
    errors=np.array([[0.]*4]*100+[[1.]*4]*100)
    plain=a.compile_joint(a.fit_joint(answers,errors,'equality',1),POOL)
    signed=a.compile_joint(a.fit_joint(answers,errors,'polarity',1),POOL)
    def score(policy):
        out=[]
        for row,e in zip(answers,errors):
            choice,_,order,*_=policy.execute(lambda i:row[i],pool=POOL)
            out.append((.25 if choice<0 else e[choice])+.01*len(order))
        return np.mean(out)
    assert score(signed)<score(plain)
    assert score(signed)==pytest.approx(.135)

@pytest.mark.parametrize('kind',['equality','polarity'])
def test_new_policy_never_emits_invalid(kind):
    responses=np.full((30,4),None,dtype=object);errors=np.zeros((30,4))
    policy=a.compile_joint(a.fit_joint(responses,errors,kind),POOL)
    assert policy.execute(lambda i:None,pool=POOL)[0]==-1

@pytest.mark.parametrize('kind',['equality','polarity'])
def test_pool_mismatch_before_acquisition(kind):
    responses,errors=data();p=a.compile_joint(a.fit_joint(responses,errors,kind),POOL)
    def forbidden(i):raise AssertionError('Must reject before querying')
    with pytest.raises(ValueError):p.execute(forbidden,pool=a.Pool(POOL.models,'version-2'))
    with pytest.raises(ValueError):p.execute(forbidden,pool=a.Pool(tuple(reversed(POOL.models)),POOL.revision))

@pytest.mark.parametrize('kind',['equality','polarity'])
def test_all_complete_paths_bounded(kind):
    responses,errors=data();p=a.compile_joint(a.fit_joint(responses,errors,kind),POOL)
    for pattern in a.graph(kind)[0]:
        seen=[]
        row=[None if x==-1 else (('no','yes')[x] if kind=='polarity' else str(x)) for x in pattern]
        def acquire(i):seen.append(i);return row[i]
        choice,risk,order,batches,state=p.execute(acquire,pool=POOL)
        assert len(seen)<=3 and len(seen)==len(set(seen))
        assert choice==-1 or choice in seen
        assert choice==-1 or row[choice] is not None

@pytest.mark.parametrize('field',['missing_state','duplicate_state','bad_query','unacquired','nan','schema','pool','cost'])
def test_serialized_fail_closed(field):
    responses,errors=data();p=a.compile_joint(a.fit_joint(responses,errors),POOL)
    obj=json.loads(p.to_json())
    if field=='missing_state':obj['states'].pop()
    if field=='duplicate_state':obj['states'].append(obj['states'][0])
    if field=='bad_query':obj['states'][0][2]=[0,0]
    if field=='unacquired':obj['states'][0][2]=[];obj['states'][0][3]=2
    if field=='nan':obj['states'][0][4]=float('nan')
    if field=='schema':obj['schema']='wrong'
    if field=='pool':obj['pool']['revision']='new'
    if field=='cost':obj['costs'][0]=-1
    with pytest.raises(ValueError):a.CompiledPolicy.from_json(json.dumps(obj),expected_pool=POOL)


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError):a.CompiledPolicy.from_json('{"schema":"a","schema":"b"}',expected_pool=POOL)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,2.])
def test_invalid_error_mass(bad):
    responses,errors=data();errors[0,0]=bad
    with pytest.raises(ValueError):a.fit_joint(responses,errors)

@pytest.mark.parametrize('bad',[0.,-1.,float('nan')])
def test_invalid_prior(bad):
    responses,errors=data()
    with pytest.raises(ValueError):a.fit_joint(responses,errors,strength=bad)


def test_leaf_prior_normalizes_mass():
    responses,errors=data();prior=a.fit_joint(responses,errors,'polarity')
    leaf=a.fit_joint(responses[:12],errors[:12],'polarity',32,prior)
    assert leaf.mass.sum()==pytest.approx(44)
    leaf.validate()


def test_semantics_not_arbitrary_binary_guess():
    with pytest.raises(ValueError):a.encode(['maybe'],'polarity')
    assert a.encode(['yes','no'],'equality')==a.encode(['no','yes'],'equality')
    assert a.encode(['yes','no'],'polarity')!=a.encode(['no','yes'],'polarity')


def context_data(n=200):
    answers=np.array([['yes']*4]*n,dtype=object)
    texts=np.array(['simple easy arithmetic problem']*(n//2)+['complex uncertain ambiguous question']*(n-n//2))
    errors=np.array([[0.]*4]*(n//2)+[[1.]*4]*(n-n//2))
    return t.Data('strategyqa',np.arange(n).astype(str),np.array([f'group-{i}' for i in range(n)]),
                  np.array(['train']*n),texts,{'replacement':answers},{'replacement':errors},{'replacement':errors})


def test_context_fits_disjoint_and_roundtrips():
    d=context_data();rep=d.subset(np.arange(0,200,2));cal=d.subset(np.arange(1,200,2))
    model=a.fit_context(rep,cal,'replacement',POOL,'polarity',32,16)
    restored=a.context_from_json(a.context_to_json(model),expected_pool=POOL)
    assert set(model.bins(['simple easy arithmetic problem','complex uncertain ambiguous question']))=={0,1}
    np.testing.assert_array_equal(model.bins(d.text),restored.bins(d.text))
    for prompt in d.text[::15]:
        x=model.execute(prompt,lambda i:'yes',pool=POOL);y=restored.execute(prompt,lambda i:'yes',pool=POOL)
        assert x[0]==y[0] and x[2:]==y[2:]


def test_context_overlap_rejected():
    d=context_data()
    with pytest.raises(ValueError):a.fit_context(d,d,'replacement',POOL)


def test_sparse_context_backoff():
    d=context_data();rep=d.subset(np.arange(0,200,2));cal=d.subset(np.array([1,3,101,103]))
    model=a.fit_context(rep,cal,'replacement',POOL,min_support=32)
    assert model.leaves[0]._records==model.leaves[1]._records


def test_context_corrupt_numeric_document():
    d=context_data();rep=d.subset(np.arange(0,200,2));cal=d.subset(np.arange(1,200,2))
    model=a.fit_context(rep,cal,'replacement',POOL);obj=json.loads(a.context_to_json(model));obj['idf'][0]=float('inf')
    with pytest.raises(ValueError):a.context_from_json(json.dumps(obj),expected_pool=POOL)


def test_gate_rejects_small_sample_overconfidence():
    bound=a.promotion_bound(np.full(10,-.01))
    assert not bound['promote'] and bound['upper_difference']>0


def test_gate_can_promote_large_stable_improvement():
    bound=a.promotion_bound(np.full(1000,-.1))
    assert bound['promote'] and bound['upper_difference']<0


def test_gate_constant_not_zero_radius():
    bound=a.promotion_bound(np.zeros(100))
    assert bound['radius']>0 and not bound['promote']

@pytest.mark.parametrize('bad',[[0],[0,float('nan')],[-2,0]])
def test_bad_gate_rejected(bad):
    with pytest.raises(ValueError):a.promotion_bound(bad)


def test_group_split_invariant_under_label_changes():
    d=context_data();d.splits[:80]='train';d.splits[80:160]='dev';d.splits[160:]='test'
    _,m=s.split_roles(d,1601);d.errors['replacement'][:]=1-d.errors['replacement']
    _,n=s.split_roles(d,1601)
    assert m.equals(n) and m.role.nunique()==5
    assert (m.groupby('group').role.nunique()==1).all()


def test_policy_does_not_see_unqueried_answers():
    responses,errors=data();old=t.Policy(t.World(responses,errors),'single');new=a.CompiledPolicy.from_legacy(old,POOL)
    first=new._records[(0,())][0]
    # `single` structurally queries worker 0 at the root, so `first` must be
    # non-empty: without this guard the test below would pass vacuously if the
    # fixture ever stopped forcing an acquisition.
    assert first, 'fixture must force at least one acquisition'
    calls=[]
    def acquire(i):
        assert i in first
        calls.append(i)
        return 'yes'
    new.execute(acquire,pool=POOL)
    assert tuple(calls)==first

@pytest.mark.parametrize('seed',[1,7])
def test_signed_policy_against_independent_recursive_enumerator(seed):
    from functools import lru_cache
    responses,errors=data(seed,50)
    joint=a.fit_joint(responses,errors,'polarity',10.)
    policy=a.compile_joint(joint,POOL)
    support=list(itertools.product((-1,0,1),repeat=4))
    @lru_cache(None)
    def solve(seen,values):
        indices=[j for j,p in enumerate(support) if tuple(p[i] for i in seen)==values]
        den=sum(joint.mass[j] for j in indices)
        best=min([.25]+[sum(joint.error_sum[j,i] for j in indices)/den for pos,i in enumerate(seen) if values[pos]>=0])
        left=[i for i in range(4) if i not in seen]
        for n in range(1,4-len(seen)):
            for batch in itertools.combinations(left,n):
                new=tuple(sorted(seen+batch));children={}
                for j in indices:
                    v=tuple(support[j][i] for i in new)
                    children[v]=children.get(v,0.)+joint.mass[j]
                best=min(best,.01*n+sum(m/den*solve(new,v) for v,m in children.items()))
        return best
    total=0.
    for j,p in enumerate(support):
        row=[None if x==-1 else ('no','yes')[x] for x in p]
        choice,_,order,*_=policy.execute(lambda i:row[i],pool=POOL)
        total+=joint.mass[j]*(.25 if choice<0 else joint.error_sum[j,choice]/joint.mass[j])+.01*len(order)*joint.mass[j]
    assert total/joint.mass.sum()==pytest.approx(solve((),()),abs=1e-12)

"""Boundary, parser, split, and independent Bellman checks for real-trace replay."""
import io,itertools,pickle
from functools import lru_cache
import numpy as np
import pandas as pd
import pytest
import external_replay as e

@pytest.mark.parametrize('text,expected',[
 ("['A']",0),("['B)']",1),("['C\\n']",2),("D.",3),('Answer: A',0),
 ('The correct answer is B.',1),('(C)',2),('[D]',3),('',4),('maybe A or B',4),
 ('All choices are possible',4),('Aardvark',4),('E',4),('A) explanation',0),
 ('[\'D)\']',3),('I think A',4),(None,4)])
def test_parser(text,expected):
    assert e.parse_answer(text)==expected

def test_no_gold_argument_in_parser():
    with pytest.raises(TypeError):e.parse_answer('A','B')

def test_prompt_group_normalization():
    assert e.group_hash("['Hello  WORLD']")==e.group_hash('hello world')
    assert e.split_group(e.group_hash('x'))==e.split_group(e.group_hash('x'))

def test_seeded_split_is_disjoint():
    groups=[e.group_hash(str(i)) for i in range(2000)]
    sets=[{g for g in groups if e.split_group(g)==s} for s in ('train','dev','test')]
    assert all(sets)
    assert all(not a&b for a,b in itertools.combinations(sets,2))

def make_fit(seed=1):
    rng=np.random.default_rng(seed);x=rng.integers(0,5,(200,4));y=rng.integers(0,2,(200,4))
    return e.fit(x,y,np.tile([.0001,.0003,.002,.005],(200,1)),10.)

def test_reject_pickle_globals():
    class Bad:
        def __reduce__(self):return (eval,('1+1',))
    with pytest.raises(pickle.UnpicklingError):e.RestrictedUnpickler(io.BytesIO(pickle.dumps(Bad()))).load()

@pytest.mark.parametrize('strength',[0,-1,np.nan,np.inf])
def test_bad_prior(strength):
    with pytest.raises(ValueError):e.fit(np.zeros((2,4),int),np.ones((2,4)),np.ones((2,4)),strength)

def test_bad_cost():
    with pytest.raises(ValueError):e.fit(np.zeros((2,4),int),np.ones((2,4)),-np.ones((2,4)),1.)

def test_unreachable_histories_finite():
    f=e.fit(np.zeros((5,4),int),np.ones((5,4)),np.ones((5,4))*.001,1.)
    assert np.isfinite(f.mean_scores).all() and (f.mass>0).all()

@pytest.mark.parametrize('name',e.POLICIES)
def test_acquisition_and_selection_boundaries(name):
    p=e.build_policy(make_fit(),name,1.,1.)
    for x in e.FULL[::13]:
        seen=[]
        def acquire(i):
            assert i not in seen
            seen.append(i);return int(x[i]),.001
        selected,cost,n,state,batches=e.execute(p,acquire)
        assert n==len(seen)<=3
        assert selected==-1 or selected in seen
        assert cost==pytest.approx(n*.001)

@pytest.mark.parametrize('name',e.POLICIES)
def test_labels_cannot_change_execution(name):
    f=make_fit();p=e.build_policy(f,name,1.,.25)
    x=np.array([[1,2,1,0],[0,0,0,0]])
    d=e.TraceData(np.array(['a','b']),np.array(['ga','gb']),np.array(['mmlu','mmlu']),
       np.array(['mmlu-test','mmlu-test']),np.array(['test','test']),x,np.zeros((2,4)),np.ones((2,4))*.002)
    before=e.evaluate(p,d,1.,.25,'test')
    d.scores[:]=1
    after=e.evaluate(p,d,1.,.25,'test')
    cols=['selected_model_index','batch_masks','terminal_state','historical_cost_usd','acquisitions']
    pd.testing.assert_frame_equal(before[cols],after[cols],check_exact=True)

def test_unqueried_observations_noninterference():
    p=e.build_policy(make_fit(),'single_best',10.,1.)
    x=np.array([0,1,2,3]);seen=[]
    def read(i):seen.append(i);return int(x[i]),.001
    original=e.execute(p,read)
    for i in set(range(4))-set(seen):x[i]=4
    assert e.execute(p,lambda i:(int(x[i]),.001))==original

def test_oracle_field_not_used():
    row={'sample_id':'mmlu-test.0','prompt':"['question']",'eval_name':'mmlu-test'}
    for m in e.MODELS:row.update({m:1,m+'|model_response':"['A']",m+'|total_cost':.001})
    df=pd.DataFrame([row]);a,au=e.prepare(df)
    df['oracle_model_to_route_to']='DO NOT USE';b,bu=e.prepare(df)
    assert np.array_equal(a.answers,b.answers) and au==bu

def test_invalid_response_retained():
    row={'sample_id':'mmlu-test.0','prompt':'question','eval_name':'mmlu-test'}
    for m in e.MODELS:row.update({m:0,m+'|model_response':'unparseable',m+'|total_cost':.001})
    data,audit=e.prepare(pd.DataFrame([row]))
    assert len(data)==1 and np.all(data.answers==4)
    assert sum(audit['invalid_answer_count_by_model'].values())==4

@pytest.mark.parametrize('seed',[2,7,11])
def test_independent_tuple_solver(seed):
    f=make_fit(seed);lam=1.;defer=.7
    # Computes history membership and conditional probabilities directly from
    # full vectors, independent of production MATCH/STATES/TRANS tables.
    @lru_cache(None)
    def solve(h):
        ix=np.array([i for i,v in enumerate(h) if v>=0],dtype=int)
        match=np.all(e.FULL[:,ix]==np.array(h)[ix],axis=1)
        mass=f.joint_mass[match].sum()
        best=defer
        if len(ix):
            correct=f.joint_score_sums[match].sum(axis=0)/mass
            best=min(best,float((1-correct[ix]).min()))
        remaining=[i for i in range(4) if h[i]<0]
        for k in range(1,min(3-len(ix),len(remaining))+1):
            for batch in itertools.combinations(remaining,k):
                q=lam*f.expected_cost[list(batch)].sum()
                for values in itertools.product(range(5),repeat=k):
                    child=list(h)
                    for i,v in zip(batch,values):child[i]=v
                    cix=[i for i,v in enumerate(child) if v>=0]
                    cm=np.all(e.FULL[:,cix]==np.array(child)[cix],axis=1)
                    q+=f.joint_mass[cm].sum()/mass*solve(tuple(child))
                best=min(best,q)
        return float(best)
    p=e.build_policy(f,'bellman',lam,defer)
    assert p.predicted_value==pytest.approx(solve((-1,)*4),abs=1e-11)

@pytest.mark.parametrize('seed',[1,2,3])
def test_fitted_dp_dominates_static_choices(seed):
    f=make_fit(seed);p=e.build_policy(f,'bellman',1.,1.)
    for name in ('single_best','fixed_three','static_all_sources'):
        assert p.predicted_value<=e.build_policy(f,name,1.,1.).predicted_value+1e-12

from external_audit import prompt_key,gold_index

def test_gold_match_requires_question_and_ordered_choices():
    p='Please answer with the letter of the correct answer.\n\nWhat?\nA) one\nB) two\nC) three\nD) four\nPrint only a single choice from ABCD'
    assert prompt_key(p,'mmlu-abstract-algebra')==('abstract_algebra','what?',('one','two','three','four'))
    assert prompt_key(p.replace('A) one','A) five'),'mmlu-abstract-algebra')!=prompt_key(p,'mmlu-abstract-algebra')

def test_ambiguous_gold_excluded():
    frame=pd.DataFrame([dict(subject='x',question='q',choices=['1','2','3','4'],answer=i) for i in (0,1)])
    idx,n=gold_index(frame);assert idx=={} and n==1

def test_gold_cannot_match_wrong_option_count():
    assert prompt_key('q\nA) one\nB) two','mmlu-test') is None

def test_reference_index_is_order_sensitive():
    f=pd.DataFrame([dict(subject='x',question='q',choices=['1','2','3','4'],answer=0)])
    idx,_=gold_index(f)
    assert ('x','q',('2','1','3','4')) not in idx

"""Independent invariants and numerical certificates for study.py."""
import itertools
from functools import lru_cache
import numpy as np
import pytest
import study as s

def independently_evaluate(policy,joint,weight,state=0):
    """Direct tree expectation; does not use optimizer values or replay lookup."""
    probabilities=s.MATCH @ joint.T
    mass=probabilities.sum(axis=1)
    _,_,charges=s.batch_terms()
    @lru_cache(None)
    def recurse(h):
        action=int(policy.actions[h])
        if not action:
            d=int(policy.decisions[h])
            return s.CFG['defer_loss'] if d==-1 else weight*probabilities[h,1-d]/mass[h]
        children=h+np.unique(s.ADD[action])
        return charges[action]+sum(mass[c]/mass[h]*recurse(int(c)) for c in children)
    return recurse(state)

@pytest.mark.parametrize('scenario',['id']+s.CFG['stress_tests'])
def test_distribution_normalized(scenario):
    js=s.make_joints(scenario)
    assert np.all(js>=0)
    np.testing.assert_allclose(js.sum(axis=(1,2)),1.,atol=1e-12)

@pytest.mark.parametrize('typ',range(5))
def test_bellman_equals_independent_tree(typ):
    j=s.make_joints()[typ]; p=s.optimize(j,s.WEIGHTS[typ])
    val=independently_evaluate(p,j,s.WEIGHTS[typ])
    assert abs(p.predicted_value-val)<1e-10
    compiled=s.compile_policy(p)
    replay=np.sum(j*s.result_vectors(compiled,s.WEIGHTS[typ])['objective'])
    assert abs(replay-val)<1e-10

@pytest.mark.parametrize('seed',range(10))
def test_optimality_bellman_certificate_random_joint(seed):
    rng=np.random.default_rng(seed)
    j=rng.dirichlet(np.full(128,.7)).reshape(2,64)
    w=float(rng.uniform(.01,20))
    p=s.optimize(j,w)
    mass,_,_,risk=s.terminal(j,w)
    _,_,charges=s.batch_terms()
    # Independently verify every accessible subproblem and all feasible batches.
    for h in np.flatnonzero(s.OBS_COUNT<=s.BUDGET):
        v=independently_evaluate(p,j,w,int(h))
        choices=[risk[h]]
        for m,ch in s.TRANS[h]:
            q=charges[m]+sum(mass[c]/mass[h]*independently_evaluate(p,j,w,int(c)) for c in ch)
            choices.append(q)
        assert abs(v-min(choices))<1e-10

@pytest.mark.parametrize('typ',range(5))
def test_oracle_dominates_controls(typ):
    js=s.make_joints(); p=s.build_policies(js)
    vals={name:independently_evaluate(ps[typ],js[typ],s.WEIGHTS[typ]) for name,ps in p.items()}
    assert vals['P1_bellman']<=min(vals.values())+1e-10

@pytest.mark.parametrize('typ',range(5))
def test_no_unqueried_information_leakage(typ):
    j=s.make_joints()[typ]; p=s.optimize(j,s.WEIGHTS[typ]); c=s.compile_policy(p)
    for o in range(64):
        mask=int(c['mask'][o])
        for alternative in range(64):
            if (alternative & mask)==(o & mask):
                assert c['decision'][alternative]==c['decision'][o]
                assert c['mask'][alternative]==c['mask'][o]
                assert c['cost'][alternative]==c['cost'][o]
                assert c['latency'][alternative]==c['latency'][o]

def test_preferences_unidentifiable_without_elicitation():
    j=s.preference_joint()
    # All possible histories that exclude E carry exactly zero information on Y.
    mass,post,_,_=s.terminal(j,8.)
    no_e=s.STATES[:,5]==0
    np.testing.assert_allclose(post[no_e],.5,atol=1e-12)
    for k in range(6):
        eq=float(sum(j[y,o] for y in range(2) for o in range(64) if s.BITS[o,k]==y))
        assert abs(eq-(.95 if k==5 else .5))<1e-12

def test_xor_complementarity():
    # A,B independent fair bits; Y=A xor B. All other channels are uninformative.
    j=np.zeros((2,64))
    for o in range(64): j[int(s.BITS[o,0]^s.BITS[o,1]),o]=1/64
    # Add a tiny full-support component for well-defined unused histories.
    j=.999999*j+.000001/128
    p=s.optimize(j,1.,cost=np.ones(6)*.12,latency=np.zeros(6),lam=1.,mu=0.)
    g=s.optimize(j,1.,kind='greedy',cost=np.ones(6)*.12,latency=np.zeros(6),lam=1.,mu=0.)
    assert abs(p.predicted_value-.2400005)<1e-9
    assert g.actions[0]==0
    assert abs(g.predicted_value-.5)<1e-12

def test_empty_low_stakes_route():
    p=s.optimize(s.preference_joint(),.05)
    assert p.actions[0]==0
    assert abs(p.predicted_value-.025)<1e-12

def test_probability_partition_all_transitions():
    j=s.make_joints()[1]; mass=(s.MATCH @ j.T).sum(axis=1)
    for h, options in s.TRANS.items():
        for m,ch in options:
            assert abs(mass[h]-mass[ch].sum())<1e-12
            assert all(s.OBS_COUNT[c]==s.OBS_COUNT[h]+m.bit_count() for c in ch)

def test_parallel_latency_not_summed():
    c,l,_=s.batch_terms()
    panel=1|4|8
    assert abs(l[panel]-max(s.LAT[[0,2,3]]))<1e-12
    assert abs(c[panel]-s.COST[[0,2,3]].sum())<1e-12

def test_repeatability():
    a=s.sample_cases(s.make_joints(),1000,np.random.default_rng(123))
    b=s.sample_cases(s.make_joints(),1000,np.random.default_rng(123))
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)

def test_diversity_counterexample():
    c=s.counterexamples()['diversity_only']
    assert abs(c['independent_weak_majority_error']-.42525)<1e-12
    assert c['independent_weak_majority_error']>c['correlated_strong_majority_error']

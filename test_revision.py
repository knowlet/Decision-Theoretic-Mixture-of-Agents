"""Independent tuple-history solver and publication regression tests."""
from functools import lru_cache
from itertools import combinations, product
import numpy as np
import pytest
import study as s
from revision import static_all


def reference_value(joint, weight, budget, cost, latency, gate=False):
    """No MATCH, TRANS, ADD, ternary states, optimizer or replay helper used.

    Enumerate posterior cells directly, then solve over tuple histories. This is
    a second implementation for numerical validation, not a formal proof.
    """
    @lru_cache(None)
    def solve(history):
        valid=[o for o in range(64) if all(((o>>i)&1)==z for i,z in history)]
        mass=float(joint[:,valid].sum())
        if mass==0:return 0.
        py=[float(joint[y,valid].sum()/mass) for y in (0,1)]
        best=min(weight*py[0],weight*py[1],s.CFG['defer_loss'])
        seen={i for i,_ in history};available=[i for i in range(6) if i not in seen]
        for k in range(1,min(len(available),budget-len(history))+1):
            for batch in combinations(available,k):
                if gate and 4 in batch and not any(i<4 for i in seen):continue
                q=.1*sum(cost[i] for i in batch)+.04*max(latency[i] for i in batch)
                for bits in product((0,1),repeat=k):
                    nxt=tuple(sorted(history+tuple(zip(batch,bits))))
                    cells=[o for o in valid if all(((o>>i)&1)==z for i,z in zip(batch,bits))]
                    prob=float(joint[:,cells].sum()/mass)
                    if prob:q+=prob*solve(nxt)
                best=min(best,q)
        return best
    return solve(())

@pytest.mark.parametrize('seed',range(6))
@pytest.mark.parametrize('gate',[False,True])
def test_independent_tuple_solver(seed,gate):
    rng=np.random.default_rng(101+seed)
    j=rng.dirichlet(np.ones(128)*.4).reshape(2,64)
    w=float(rng.uniform(.1,20));cost=rng.uniform(.1,2,6);lat=rng.uniform(.1,3,6)
    p=s.optimize(j,w,budget=2,cost=cost,latency=lat,candidate_required=gate)
    assert abs(p.predicted_value-reference_value(j,w,2,cost,lat,gate))<1e-10

@pytest.mark.parametrize('budget',[0,1,2,3,4])
def test_budget_and_gate(budget):
    p=s.optimize(s.make_joints()[2],20.,budget=budget,candidate_required=True)
    c=s.compile_policy(p)
    assert max(c['calls'])<=budget
    for trace in c['histories']:
        seen=0
        for mask in trace:
            assert not(mask&seen)
            if mask&16:assert seen&15
            seen|=mask

@pytest.mark.parametrize('seed',range(5))
def test_static_and_greedy_are_feasible_restrictions(seed):
    rng=np.random.default_rng(seed+300)
    j=rng.dirichlet(np.ones(128)).reshape(2,64)
    w=10.
    dp=s.optimize(j,w);static=static_all(j,w)
    greedy=s.optimize(j,w,'greedy')
    assert dp.predicted_value<=static.predicted_value+1e-10
    assert dp.predicted_value<=greedy.predicted_value+1e-10

@pytest.mark.parametrize('bad',[np.ones((2,63))/126, np.full((2,64),np.nan),
                                np.ones((2,64)), np.full((2,64),-1/128)])
def test_invalid_distribution_rejected(bad):
    with pytest.raises(ValueError):s.optimize(bad,1.)

@pytest.mark.parametrize('options',[{'kind':'typo'},{'budget':-1},{'budget':5},
                                    {'allowed':64},{'lam':-1},{'mu':float('nan')},
                                    {'max_batch':0},{'cost':np.ones(5)},
                                    {'latency':np.full(6,-1.)}])
def test_invalid_optimizer_arguments_rejected(options):
    with pytest.raises(ValueError):s.optimize(s.make_joints()[0],1.,**options)

def test_sparse_support_without_fake_smoothing():
    j=np.zeros((2,64));j[0,0]=.5;j[1,63]=.5
    with np.errstate(divide='raise',invalid='raise'):
        p=s.optimize(j,1.)
    assert np.isfinite(p.posterior).all()
    # Cheapest perfect signal D costs .055 + .026.
    assert abs(p.predicted_value-.081)<1e-12
    assert abs(reference_value(j,1.,2,s.COST,s.LAT)-.081)<1e-12

def test_defer_from_configuration(monkeypatch):
    monkeypatch.setitem(s.CFG,'defer_loss',.123)
    p=s.optimize(s.preference_joint(),8.,budget=0)
    assert p.decisions[0]==-1
    assert abs(p.predicted_value-.123)<1e-12
    value=s.result_vectors(s.compile_policy(p),8.)['objective']
    np.testing.assert_allclose(value,.123)

@pytest.mark.parametrize('seed',range(5))
def test_estimation_plus_optimization_regret_bound(seed):
    rng=np.random.default_rng(seed+900)
    truth=s.factual_joint();fit=s.calibrate(truth,200,rng);w=5.
    approximate=s.optimize(fit,w,'greedy')
    fit_opt=s.optimize(fit,w);true_opt=s.optimize(truth,w)
    c=s.compile_policy(approximate)
    l=s.result_vectors(c,w)['objective']
    delta=float(np.sum(fit*l))-fit_opt.predicted_value
    actual=float(np.sum(truth*l))-true_opt.predicted_value
    tv=.5*np.abs(fit-truth).sum()
    M=max(w,s.CFG['defer_loss'])+.1*sum(sorted(s.COST)[-4:])+.04*sum(sorted(s.LAT)[-4:])
    assert delta>=-1e-10
    assert actual<=2*M*tv+delta+1e-10

@pytest.mark.parametrize('accuracy',[.5,.6,.75,.9,.99,1.])
def test_total_variation_preference_bound(accuracy):
    j=s.preference_joint(accuracy)
    m=np.array([[j[y,s.BITS[:,5]==z].sum() for z in (0,1)] for y in (0,1)])
    tv=.5*np.abs(2*m[0]-2*m[1]).sum()
    assert abs(m.max(axis=0).sum()-(1+tv)/2)<1e-12

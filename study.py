"""Finite-support MoA simulator. No model APIs and no language-model claims.

Run: python study.py --output results
Requires Python >=3.10, numpy, scipy, pandas, matplotlib.
The joint distribution preserves correlated failures. Policies never see Y or
unqueried observations. Calibration and test use disjoint RNG streams.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, platform, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "protocol.json").read_text())
N = 6
BITS = ((np.arange(2**N)[:, None] >> np.arange(N)) & 1).astype(np.int8)
P3 = 3**np.arange(N)
STATES = ((np.arange(3**N)[:, None] // P3) % 3).astype(np.int8)
OBS_COUNT = (STATES > 0).sum(axis=1)
MASKS = np.arange(2**N)
MASK_BITS = ((MASKS[:, None] >> np.arange(N)) & 1).astype(bool)
MATCH = np.all((STATES[:, None, :] == 0) |
               (STATES[:, None, :] == BITS[None, :, :] + 1), axis=2).astype(float)
ADD = (MASK_BITS.astype(np.int16) * P3) @ (BITS + 1).T
COST = np.array(CFG["cost"], dtype=float)
LAT = np.array(CFG["latency"], dtype=float)
BUDGET = CFG["max_acquisitions"]
WEIGHTS = np.array(CFG["wrong_decision_loss"])
PROBS = np.array(CFG["task_probabilities"])
TYPE_NAMES = CFG["task_types"]
ALL_ALLOWED = (1 << N) - 1

# For each history, all allowed batches and all possible child histories.
TRANS: dict[int, list[tuple[int, np.ndarray]]] = {}
for s in range(3**N):
    if OBS_COUNT[s] >= BUDGET:
        TRANS[s] = []
        continue
    observed = int(sum((1 << i) for i in range(N) if STATES[s, i]))
    opts = []
    for m in range(1, 1 << N):
        if m & observed or m.bit_count() > BUDGET - OBS_COUNT[s]:
            continue
        children = s + np.unique(ADD[m])
        opts.append((m, children.astype(np.int16)))
    TRANS[s] = opts
ORDER = sorted(range(3**N), key=lambda s: int(OBS_COUNT[s]), reverse=True)

@dataclass
class Policy:
    actions: np.ndarray   # 0 means stop; otherwise a bitmask batch
    decisions: np.ndarray  # 0,1 or -1 (defer)
    predicted_value: float
    posterior: np.ndarray

def batch_terms(cost=COST, latency=LAT, lam=None, mu=None):
    lam = CFG["lambda_cost"] if lam is None else lam
    mu = CFG["mu_latency"] if mu is None else mu
    c = MASK_BITS @ cost
    l = np.max(MASK_BITS * latency[None, :], axis=1)
    return c, l, lam*c + mu*l

def factual_joint(easy=False, global_error=None, verifier_accuracy=0.96,
                  verifier_coupling=0.0, perfect_easy=False):
    """Y~Bern(.5). Shared global and A/B family flips cause co-failures."""
    if perfect_easy:
        individual = np.array([.995, .88, .85, .8])
        g, f = 0.0, 0.0
    elif easy:
        individual = np.array([.95, .92, .90, .86])
        g, f = .015, .035
    else:
        individual = np.array([.82, .80, .77, .74])
        g, f = .12, .08
    if global_error is not None:
        g = global_error
    out = np.zeros((2, 64))
    for y, global_flip, family_flip in itertools.product((0,1), repeat=3):
        zprob = .5 * (g if global_flip else 1-g) * (f if family_flip else 1-f)
        p1 = np.empty(6)
        for a in range(4):
            target = y ^ global_flip ^ (family_flip if a < 2 else 0)
            p1[a] = individual[a] if target else 1-individual[a]
        # Optional verifier common-mode failure; zero in the main condition.
        target_v = y ^ global_flip
        base = verifier_accuracy if y else 1-verifier_accuracy
        shared = verifier_accuracy if target_v else 1-verifier_accuracy
        p1[4] = (1-verifier_coupling)*base + verifier_coupling*shared
        p1[5] = .5  # eliciting preferences does not identify factual truth
        out[y] += zprob * np.prod(np.where(BITS, p1, 1-p1), axis=1)
    return out / out.sum()

def preference_joint(elicitor_accuracy=.95):
    """A user's private utility-optimal choice Y is independent of agent taste Z.

    Agents A-D share a population-style preference Z, and thus agree often,
    but their opinions do not identify this user's utility. Only E observes Y.
    """
    out = np.zeros((2,64))
    acc = np.array([.92,.90,.85,.80])
    for y,z in itertools.product((0,1), repeat=2):
        p1 = np.empty(6)
        p1[:4] = acc if z else 1-acc
        p1[4] = .5
        p1[5] = elicitor_accuracy if y else 1-elicitor_accuracy
        out[y] += .5*(.7 if z else .3)*np.prod(np.where(BITS,p1,1-p1),axis=1)
    return out / out.sum()

def make_joints(scenario="id"):
    kwargs = {}
    if scenario == "common_failure_shift": kwargs["global_error"] = .32
    if scenario == "verifier_degradation": kwargs["verifier_accuracy"] = .62
    eacc = .55 if scenario == "elicitor_degradation" else .95
    return np.array([factual_joint(easy=True, **kwargs),
                     factual_joint(**kwargs), factual_joint(**kwargs),
                     preference_joint(eacc), preference_joint(eacc)])

def calibrate(joint, n, rng):
    # Half-count smoothing in each of 128 cells. No test labels are used.
    count = rng.multinomial(n, joint.ravel())
    alpha = CFG["dirichlet_pseudocount_per_joint_cell"]
    return ((count + alpha)/(n + alpha*count.size)).reshape(2,64)

def independent_approx(joint):
    out = np.empty_like(joint)
    py = joint.sum(axis=1)
    for y in (0,1):
        p1 = (joint[y] @ BITS) / py[y]
        out[y] = py[y]*np.prod(np.where(BITS,p1,1-p1),axis=1)
    return out/out.sum()

def validate_joint(joint):
    joint = np.asarray(joint, dtype=float)
    if joint.shape != (2, 64) or not np.all(np.isfinite(joint)):
        raise ValueError("joint must be a finite (2, 64) probability table")
    if np.any(joint < 0) or not np.isclose(joint.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("joint probabilities must be nonnegative and sum to one")
    return joint

def terminal(joint, weight, defer=None):
    joint = validate_joint(joint)
    defer = CFG["defer_loss"] if defer is None else defer
    if not np.isfinite(weight) or weight < 0 or not np.isfinite(defer) or defer < 0:
        raise ValueError("terminal losses must be finite and nonnegative")
    masses = MATCH @ joint.T
    mass = masses.sum(axis=1)
    # Unreachable histories have an arbitrary, finite posterior and stop.
    # Their mass is zero and cannot affect any reachable expectation.
    post = np.divide(masses[:,1], mass, out=np.full_like(mass, .5), where=mass > 0)
    loss = np.column_stack([weight*post, weight*(1-post), np.full(len(post),defer)])
    choice = np.argmin(loss,axis=1)
    decisions = np.where(choice==2,-1,choice).astype(np.int8)
    risks = loss[np.arange(len(post)),choice]
    return mass,post,decisions,risks

def optimize(joint, weight, kind="dp", allowed=ALL_ALLOWED, force_budget=False,
             cost=COST, latency=LAT, lam=None, mu=None, budget=BUDGET,
             candidate_required=False, max_batch=None):
    if kind not in ("dp", "greedy", "greedy_batch"):
        raise ValueError("kind must be dp, greedy, or greedy_batch")
    if not isinstance(budget, int) or not 0 <= budget <= BUDGET:
        raise ValueError(f"budget must be an integer from 0 to {BUDGET}")
    if not isinstance(allowed, int) or allowed < 0 or allowed & ~ALL_ALLOWED:
        raise ValueError("allowed must be a valid channel bitmask")
    if max_batch is not None and (not isinstance(max_batch, int) or max_batch < 1):
        raise ValueError("max_batch must be a positive integer")
    for values in (cost, latency):
        a = np.asarray(values)
        if a.shape != (N,) or np.any(~np.isfinite(a)) or np.any(a < 0):
            raise ValueError("resource vectors must be finite, nonnegative and length six")
    for v in (lam, mu):
        if v is not None and (not np.isfinite(v) or v < 0):
            raise ValueError("resource weights must be finite and nonnegative")
    mass,post,decisions,risk = terminal(joint,weight)
    _,_,charges = batch_terms(cost,latency,lam,mu)
    value = risk.copy()
    actions = np.zeros(3**N,dtype=np.int16)
    for s in ORDER:
        if OBS_COUNT[s] >= budget or mass[s] == 0: continue
        candidates = [(m,ch) for m,ch in TRANS[s]
                      if m & ~allowed == 0 and OBS_COUNT[s]+m.bit_count() <= budget
                      and (max_batch is None or m.bit_count() <= max_batch)
                      and not (candidate_required and m & 16 and not np.any(STATES[s,:4]))]
        if kind == "greedy":
            candidates = [(m,ch) for m,ch in candidates if m.bit_count()==1]
        if not candidates: continue
        best = math.inf if force_budget else risk[s]
        best_action = 0
        for m,ch in candidates:
            future = value[ch] if kind == "dp" else risk[ch]
            q = charges[m] + np.dot(mass[ch],future)/mass[s]
            if q < best - 1e-12:
                best,best_action = float(q),m
        if best_action:
            actions[s] = best_action
            # Store actual sequential value of the chosen policy (greedy can replan).
            ch = next(ch for m,ch in candidates if m==best_action)
            value[s] = charges[best_action] + np.dot(mass[ch],value[ch])/mass[s]
        else:
            value[s] = risk[s]
    return Policy(actions,decisions,float(value[0]),post)

def fixed_policy(joint,weight,k):
    """Select a worker subset separately for this observed task type, not globally."""
    mass,post,decisions,risk = terminal(joint,weight)
    _,_,charges = batch_terms()
    candidates = [m for m in range(1,16) if m.bit_count()==k]
    q = []
    for m in candidates:
        ch = np.unique(ADD[m])
        q.append(charges[m]+np.dot(mass[ch],risk[ch]))
    best = candidates[int(np.argmin(q))]
    actions = np.zeros(3**N,dtype=np.int16); actions[0]=best
    return Policy(actions,decisions,float(min(q)),post)

def heuristic(joint,weight,typ):
    """One explicit, reproducible instantiation of the earlier informal proposal.

    Low stakes: immediate default. Easy factual: A only.
    Other tasks: diverse A,C,D panel, then V on disagreement (factual),
    or E on preferences. Bayes terminal choice/defer is shared by all policies.
    """
    mass,post,decisions,risk = terminal(joint,weight)
    actions=np.zeros(3**N,dtype=np.int16)
    if typ == 3:
        decisions[0]=0
    elif typ == 0:
        actions[0]=1
    else:
        panel=(1<<0)|(1<<2)|(1<<3)
        actions[0]=panel
        for s in np.unique(ADD[panel]):
            disagree=len(set(STATES[s,[0,2,3]].tolist()))>1
            if typ==4: actions[s]=(1<<5)
            elif disagree: actions[s]=(1<<4)
    # Predicted risk is evaluated by complete-support replay below.
    return Policy(actions,decisions,float("nan"),post)

def compile_policy(p:Policy,cost=COST,latency=LAT):
    """Map the 64 complete potential-outcome vectors to observable policy traces.

    Replay advances the history only using outcomes of acquired channels.
    Nonacquired signals never influence a decision.
    """
    bc,bl,_ = batch_terms(cost,latency)
    d=np.zeros(64,dtype=np.int8); c=np.zeros(64); l=np.zeros(64)
    calls=np.zeros(64,dtype=np.int8); batches=np.zeros(64,dtype=np.int8)
    mask=np.zeros(64,dtype=np.int16); posterior=np.zeros(64)
    histories=[]
    for o in range(64):
        s=0; h=[]
        for _ in range(BUDGET+1):
            m=int(p.actions[s])
            if not m: break
            assert not (mask[o]&m), "A channel was queried twice"
            c[o]+=bc[m]; l[o]+=bl[m]; calls[o]+=m.bit_count(); batches[o]+=1
            mask[o]|=m; h.append(m); s+=int(ADD[m,o])
        else:
            raise AssertionError("Nonterminating policy")
        assert calls[o]<=BUDGET
        d[o]=p.decisions[s]; posterior[o]=p.posterior[s]; histories.append(h)
    return {"decision":d,"cost":c,"latency":l,"calls":calls,
            "batches":batches,"mask":mask,"posterior":posterior,"histories":histories}

def result_vectors(compiled,weight,lam=None,mu=None):
    lam=CFG["lambda_cost"] if lam is None else lam
    mu=CFG["mu_latency"] if mu is None else mu
    d=compiled["decision"]; accepted=d>=0
    err=(d[None,:]!=np.arange(2)[:,None]) & accepted
    terminal_loss=np.where(accepted,weight*err,CFG["defer_loss"])
    score=terminal_loss+lam*compiled["cost"]+mu*compiled["latency"]
    confidence=np.where(d==1,compiled["posterior"],1-compiled["posterior"])
    return {"objective":score,"terminal_loss":terminal_loss,
            "error":err.astype(float),
            "coverage":np.broadcast_to(accepted,(2,64)).astype(float),
            "unsafe90":err*(confidence>=.90),
            **{k:np.broadcast_to(compiled[k],(2,64)) for k in ("cost","latency","calls","batches")}}

def exact_metrics(joints,compiled,probs=PROBS,lam=None,mu=None):
    out={}
    for t in range(5):
        vals=result_vectors(compiled[t],WEIGHTS[t],lam,mu)
        for key,v in vals.items(): out[key]=out.get(key,0.)+probs[t]*np.sum(joints[t]*v)
    out["selective_accuracy"] = 1-out["error"]/out["coverage"] if out["coverage"] else np.nan
    return out

def build_policies(joints,all_policies=True):
    names=["C1_single","C2_fixed3","C3_greedy","P0_heuristic","P1_bellman"]
    if all_policies:
        names += ["P1_no_verifier","P1_no_elicitor","P1_no_diverse",
                  "P1_independence","P1_force_budget","C3_greedy_batch"]
    out={name:[] for name in names}
    for t in range(5):
        j=joints[t]; w=WEIGHTS[t]
        out["C1_single"].append(fixed_policy(j,w,1))
        out["C2_fixed3"].append(fixed_policy(j,w,3))
        out["C3_greedy"].append(optimize(j,w,"greedy"))
        out["P0_heuristic"].append(heuristic(j,w,t))
        out["P1_bellman"].append(optimize(j,w))
        if all_policies:
            out["P1_no_verifier"].append(optimize(j,w,allowed=ALL_ALLOWED^(1<<4)))
            out["P1_no_elicitor"].append(optimize(j,w,allowed=ALL_ALLOWED^(1<<5)))
            out["P1_no_diverse"].append(optimize(j,w,allowed=ALL_ALLOWED^(1<<2)^(1<<3)))
            out["P1_independence"].append(optimize(independent_approx(j),w))
            out["P1_force_budget"].append(optimize(j,w,force_budget=True))
            out["C3_greedy_batch"].append(optimize(j,w,"greedy_batch"))
    return out

def sample_cases(joints,n,rng,probs=PROBS):
    ts=rng.choice(5,n,p=probs).astype(np.int8)
    cells=np.empty(n,dtype=np.int16)
    for t in range(5):
        idx=np.flatnonzero(ts==t)
        cells[idx]=rng.choice(128,len(idx),p=joints[t].ravel())
    return ts,(cells//64).astype(np.int8),(cells%64).astype(np.int8)

def replicate_t_ci(a,confidence=.95):
    # Independent simulation replicate is the inference unit, not each row.
    a=np.asarray(a,float); m=float(np.mean(a))
    se=float(np.std(a,ddof=1)/math.sqrt(len(a)))
    delta=float(stats.t.ppf((1+confidence)/2,len(a)-1)*se)
    return m,m-delta,m+delta

def experiment(out:Path, reps:int):
    out.mkdir(parents=True,exist_ok=True); (out/"raw").mkdir(exist_ok=True)
    start=time.perf_counter()
    truth=make_joints()
    np.savez_compressed(out/"true_joint.npz", joint=truth)
    all_rows=[]; exact_rows=[]; category_rows=[]; stress_rows=[]; bounds=[]; samples={}
    snapshots={}
    for rep in range(reps):
        train_rng=np.random.default_rng(np.random.SeedSequence([CFG["seed_base"],rep,0]))
        test_rng=np.random.default_rng(np.random.SeedSequence([CFG["seed_base"],rep,1]))
        fitted=np.array([calibrate(j,CFG["calibration_cases_per_type"],train_rng) for j in truth])
        policies=build_policies(fitted)
        comp={name:[compile_policy(p) for p in ps] for name,ps in policies.items()}
        ts,ys,os=sample_cases(truth,CFG["test_cases_per_replicate"],test_rng)
        np.savez_compressed(out/"raw"/f"replicate_{rep:02d}.npz",type=ts,y=ys,outcome=os,fitted_joint=fitted)
        for name,cs in comp.items():
            matrices={k:np.stack([result_vectors(cs[t],WEIGHTS[t])[k] for t in range(5)])
                      for k in result_vectors(cs[0],WEIGHTS[0])}
            metrics={k:float(v[ts,ys,os].mean()) for k,v in matrices.items()}
            metrics["selective_accuracy"]=1-metrics["error"]/metrics["coverage"] if metrics["coverage"] else np.nan
            all_rows.append({"replicate":rep,"policy":name,**metrics})
            exact_rows.append({"replicate":rep,"policy":name,**exact_metrics(truth,cs)})
            for t in range(5):
                ids=ts==t
                mets={k:float(v[t,ys[ids],os[ids]].mean()) for k,v in matrices.items()}
                category_rows.append({"replicate":rep,"policy":name,"type":TYPE_NAMES[t],**mets})
        for scenario in CFG["stress_tests"]:
            target=make_joints(scenario)
            actual_cost=COST.copy(); actual_lat=LAT.copy(); probs=PROBS
            if scenario=="cost_shift":
                actual_cost[4]*=8; actual_cost[5]*=4; actual_lat[5]*=5
            if scenario=="easy_only_shift":
                target[0]=factual_joint(perfect_easy=True)
                probs=np.array([1.,0,0,0,0])
            for name in ("C1_single","C2_fixed3","C3_greedy","P0_heuristic","P1_bellman"):
                cs=[compile_policy(p,actual_cost,actual_lat) for p in policies[name]]
                stress_rows.append({"replicate":rep,"scenario":scenario,"policy":name,
                                    **exact_metrics(target,cs,probs)})
        # Finite-support TV bound for the fitted Bellman policy.
        for t in range(5):
            oracle=optimize(truth[t],WEIGHTS[t])
            oc=compile_policy(oracle)
            true_opt=float(np.sum(truth[t]*result_vectors(oc,WEIGHTS[t])["objective"]))
            fitted_true=float(np.sum(truth[t]*result_vectors(comp["P1_bellman"][t],WEIGHTS[t])["objective"]))
            tv=.5*float(np.abs(fitted[t]-truth[t]).sum())
            maxloss=max(float(WEIGHTS[t]),CFG["defer_loss"])+CFG["lambda_cost"]*sum(sorted(COST)[-BUDGET:])+CFG["mu_latency"]*sum(sorted(LAT)[-BUDGET:])
            assert fitted_true-true_opt>=-1e-10
            assert fitted_true-true_opt<=2*maxloss*tv+1e-10
            bounds.append({"replicate":rep,"type":TYPE_NAMES[t],"tv":tv,"regret":fitted_true-true_opt,"bound":2*maxloss*tv})
        if rep==0:
            snapshots={name:{TYPE_NAMES[t]:{"first_batch":int(p.actions[0]),
                                         "decision_at_empty":int(p.decisions[0]),
                                         "traces":comp[name][t]["histories"]}
                             for t,p in enumerate(ps)} for name,ps in policies.items()}
            samples={name:{TYPE_NAMES[t]:{"decisions":cs[t]["decision"].tolist(),
                                        "cost":cs[t]["cost"].tolist(),
                                        "latency":cs[t]["latency"].tolist(),
                                        "mask":cs[t]["mask"].tolist()}
                           for t in range(5)} for name,cs in comp.items()}
        print(f"completed replicate {rep+1}/{reps}",flush=True)
    df=pd.DataFrame(all_rows); df.to_csv(out/"metrics_by_replicate.csv",index=False)
    ex=pd.DataFrame(exact_rows); ex.to_csv(out/"exact_expectations.csv",index=False)
    cat=pd.DataFrame(category_rows); cat.to_csv(out/"metrics_by_type.csv",index=False)
    st=pd.DataFrame(stress_rows); st.to_csv(out/"stress_exact.csv",index=False)
    pd.DataFrame(bounds).to_csv(out/"estimation_bounds.csv",index=False)
    (out/"policy_snapshots.json").write_text(json.dumps(snapshots,indent=2))
    (out/"replay_lookup_rep0.json").write_text(json.dumps(samples,indent=2))
    rows=[]
    for name,sub in df.groupby("policy",sort=False):
        row={"policy":name}
        for k in [c for c in df.columns if c not in ("policy","replicate")]:
            m,lo,hi=replicate_t_ci(sub[k]); row.update({k:m,k+"_lo":lo,k+"_hi":hi})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/"summary.csv",index=False)
    pairs=[("P0_heuristic","C1_single"),("P0_heuristic","C2_fixed3"),
           ("P0_heuristic","C3_greedy"),("P1_bellman","C3_greedy")]
    piv=df.pivot(index="replicate",columns="policy",values="objective")
    comparisons=[]
    for a,b in pairs:
        d=piv[a]-piv[b]; m,lo,hi=replicate_t_ci(d)
        p=float(stats.ttest_1samp(d,0).pvalue)
        comparisons.append({"a":a,"b":b,"difference":m,"lo":lo,"hi":hi,"p":p})
    idx=sorted(range(len(comparisons)),key=lambda i:comparisons[i]["p"])
    last=0
    for rank,i in enumerate(idx):
        last=max(last,min(1.,(len(idx)-rank)*comparisons[i]["p"]))
        comparisons[i]["holm_p"]=last
    pd.DataFrame(comparisons).to_csv(out/"paired_comparisons.csv",index=False)
    # True-distribution policies: benchmark lower bound, not a deployed method.
    oracle_rows=[]
    oracle_pol=build_policies(truth)
    for name,ps in oracle_pol.items():
        oracle_rows.append({"policy":name,**exact_metrics(truth,[compile_policy(p) for p in ps])})
    pd.DataFrame(oracle_rows).to_csv(out/"oracle_known_distribution.csv",index=False)
    # Cost/latency sweep uses known likelihoods and exact expectations only.
    sweeps=[]
    for lam in CFG["sweeps"]["lambda_cost"]:
        for mu in CFG["sweeps"]["mu_latency"]:
            for kind,label in (("dp","P1_oracle"),("greedy","C3_oracle"),("greedy_batch","C3_batch_oracle")):
                ps=[optimize(truth[t],WEIGHTS[t],kind,lam=lam,mu=mu) for t in range(5)]
                sweeps.append({"lambda":lam,"mu":mu,"policy":label,
                               **exact_metrics(truth,[compile_policy(p) for p in ps],lam=lam,mu=mu)})
    pd.DataFrame(sweeps).to_csv(out/"cost_latency_sweep.csv",index=False)
    # Factual error correlation and all-wrong agreement probability, exactly.
    corrs=[]
    for t in range(3):
        j=truth[t]; err=(BITS[None,:,:4]!=np.arange(2)[:,None,None]).astype(float)
        means=(j[:,:,None]*err).sum(axis=(0,1))
        for a,b in itertools.combinations(range(4),2):
            cov=(j*err[:,:,a]*err[:,:,b]).sum()-means[a]*means[b]
            corr=cov/np.sqrt(means[a]*(1-means[a])*means[b]*(1-means[b]))
            corrs.append({"type":TYPE_NAMES[t],"a":CFG["channels"][a],"b":CFG["channels"][b],"correlation":corr})
    pd.DataFrame(corrs).to_csv(out/"error_correlations.csv",index=False)
    # Independent exact counterexamples, outside the primary benchmark.
    counters=counterexamples()
    (out/"counterexamples.json").write_text(json.dumps(counters,indent=2))
    env={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
         "scipy":__import__('scipy').__version__,"pandas":pd.__version__,
         "replicates":reps,"test_cases":reps*CFG["test_cases_per_replicate"],
         "calibration_cases":reps*5*CFG["calibration_cases_per_type"],
         "elapsed_seconds":time.perf_counter()-start,
         "llm_api_calls":0,"physical_gpu_used":False,
         "protocol_sha256":hashlib.sha256((ROOT/"protocol.json").read_bytes()).hexdigest()}
    (out/"run_metadata.json").write_text(json.dumps(env,indent=2))
    print(pd.DataFrame(rows)[["policy","objective","terminal_loss","cost","latency","coverage"]].to_string(index=False))
    print(json.dumps(comparisons,indent=2))

def counterexamples():
    # Bernoulli votes, exact majority error. Pairwise diversity isn't quality.
    def majority_error(acc):
        return sum(np.prod([p if z else 1-p for p,z in zip(acc,b)])
                   for b in itertools.product((0,1),repeat=3) if sum(b)<2)
    # Completely correlated A,B,C are correct with .9; independent weak votes .55.
    diverse_weak=majority_error([.55,.55,.55])
    # XOR: each individual observation has zero mutual information with truth.
    # Two queries cost .12 each, joint observation identifies truth perfectly.
    # These costs are already in the objective unit, not API prices.
    xor={"stop_risk":.5,"single_query_risk":.62,"joint_query_risk":.24,
         "single_step_greedy_regret":.26}
    return {
       "mandatory_fanout": {"perfect_single_loss":.1,"same_quality_three_calls_loss":.3,
                            "strict_excess":.2},
       "diversity_only": {"correlated_strong_majority_error":.1,
                          "independent_weak_majority_error":diverse_weak},
       "private_preferences": {"any_number_of_uninformative_agents_accuracy":.5,
                               "information_bits_about_private_preference":0.},
       "xor_complementarity": xor,
       "interpretation":"Constructed mathematical instances, not sampled LLM behavior."}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=ROOT/"results")
    ap.add_argument("--replicates",type=int,default=CFG["replicates"])
    args=ap.parse_args()
    if args.replicates<2: raise SystemExit("At least two replicates required for intervals")
    experiment(args.output,args.replicates)

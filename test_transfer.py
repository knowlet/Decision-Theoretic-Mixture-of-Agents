import itertools,math,re
import numpy as np
import pandas as pd
import pytest
import transfer_study as t
import decision_contract as d

@pytest.mark.parametrize('value,dataset,expected',[
    ('18.0','gsm8k','18/1'),('1,000','gsm8k','1000/1'),('1/2','svamp','1/2'),('.5','svamp','1/2'),
    ('1e3','svamp','1000/1'),('1/0','gsm8k',None),('Answer: 18','gsm8k',None),('NaN','gsm8k',None),
    ('yes','strategyqa','yes'),('False','strategyqa','no'),('maybe','strategyqa',None),('b','mmlu','B'),
    ('C. text','mmlu',None),('3.0','dices','3'),('6','dices',None),('nan','dices',None),
])
def test_response_normalization(value,dataset,expected):assert t.normalize_answer(value,dataset)==expected

@pytest.mark.parametrize('value,dataset,expected',[
    ('reasoning 41\n#### 42','gsm8k','42/1'),('42','gsm8k',None),('2','mmlu','C'),
    ('3.5','mmlu',None),('True','strategyqa','yes'),('-0.25','dices',None),
])
def test_gold_separate(value,dataset,expected):assert t.normalize_gold(value,dataset)==expected

def test_label_symmetry_and_invalid():
    assert t.canonical(['cat','dog','cat',None])==t.canonical([13,29,13,None])==(0,1,0,-1)
    assert len(set(t.PATTERNS))==52

def test_projection_does_not_observe_future():
    assert t.project((0,1,0,-1),3)==t.project((0,1,2,3),3)

def test_duplicate_group_split():
    assert t.group_id(' Hello\n WORLD ')==t.group_id('hello world')
    for _ in range(5): assert t.split_id(t.group_id('hello world'))==t.split_id(t.group_id(' Hello\n WORLD'))

def fixture(n=50):
    rng=np.random.default_rng(13)
    a=rng.integers(0,3,(n,4)).astype(object);gold=rng.integers(0,3,n)
    return a,(a!=gold[:,None]).astype(float)

@pytest.mark.parametrize('strength',[1.,10.,100.])
def test_transition_mass(strength):
    a,e=fixture();w=t.World(a,e,strength)
    for state in t.STATES:
        for batch in t.LEGAL[state]:
            tr=w.transitions(state,batch)
            assert sum(p for _,p in tr)==pytest.approx(1.)
            assert all(ch[0].bit_count()==state[0].bit_count()+len(batch) for ch,_ in tr)

@pytest.mark.parametrize('price',[0.,.01,.1,.6])
def test_bellman_dominates_same_fitted_model(price):
    a,e=fixture();w=t.World(a,e)
    dp=t.Policy(w,'bellman',price)
    for mode in ('single','fixed3','static','myopic'):
        p=t.Policy(w,mode,price)
        assert dp.values[t.ROOT_STATE]<=p.values[t.ROOT_STATE]+1e-12

@pytest.mark.parametrize('seed',range(3))
def test_independent_recursive_reference(seed):
    """Direct support filtering, not production MATCH/CHILDREN transition tables."""
    from functools import lru_cache
    rng=np.random.default_rng(seed);a=rng.integers(0,2,(17,4));gold=rng.integers(0,2,17);e=(a!=gold[:,None]).astype(float)
    strength=10.;price=.01;prior_error=(e.sum(0)+.5)/(len(e)+1)
    support=list(t.PATTERNS)+[t.canonical(row) for row in a]
    weights=[strength/len(t.PATTERNS)]*len(t.PATTERNS)+[1.]*len(a)
    losses=[prior_error]*len(t.PATTERNS)+list(e)
    def projection(p,seen):
        table={};ans=[]
        for i in seen:
            v=p[i]
            if v<0:ans.append(-1)
            else:
                if v not in table:table[v]=len(table)
                ans.append(table[v])
        return tuple(ans)
    @lru_cache(None)
    def ref(seen,values):
        ix=[j for j,p in enumerate(support) if projection(p,seen)==values];den=sum(weights[j] for j in ix)
        best=min([.25]+[sum(weights[j]*losses[j][i] for j in ix)/den for i in seen])
        remaining=[i for i in range(4) if i not in seen]
        for k in range(1,4-len(seen)):
            for batch in itertools.combinations(remaining,k):
                new=tuple(sorted(seen+batch));parts={}
                for j in ix:
                    v=projection(support[j],new);parts[v]=parts.get(v,0)+weights[j]
                q=price*k+sum(m/den*ref(new,v) for v,m in parts.items())
                best=min(best,q)
        return best
    got=t.Policy(t.World(a,e,strength),'bellman',price).values[t.ROOT_STATE]
    assert got==pytest.approx(ref((),()),abs=1e-12)

@pytest.mark.parametrize('mode',t.MODES)
def test_callback_boundary_budget_and_terminal(mode):
    a,e=fixture();p=t.Policy(t.World(a,e),mode)
    for row in a:
        used=[]
        def acquire(i):used.append(i);return row[i]
        choice,risk,seen,batches,state=p.execute(acquire,np.array([.1,.2,.1,.15]))
        assert len(used)<=3 and len(set(used))==len(used)
        assert choice==-1 or choice in used
        assert used==list(seen)

def test_test_gold_is_evaluator_only():
    a,e=fixture();p=t.Policy(t.World(a,e),'bellman')
    row=a[0].copy()
    one=p.execute(lambda i:row[i]);changed_labels=1-e[0]
    two=p.execute(lambda i:row[i])
    assert one==two and len(changed_labels)==4

@pytest.mark.parametrize('bad',[float('nan'),-1.,2.])
def test_invalid_errors_fail(bad):
    a,e=fixture();e[0,0]=bad
    with pytest.raises(ValueError):t.World(a,e)

@pytest.mark.parametrize('bad',[float('nan'),0.,-1.])
def test_invalid_prior_fail(bad):
    a,e=fixture()
    with pytest.raises(ValueError):t.World(a,e,bad)

def test_high_cost_stops():
    a=np.array([[0]*4,[1]*4],object);e=np.zeros((2,4))
    assert t.Policy(t.World(a,e),'bellman',price=1).execute(lambda i:0)[0]==-1

def test_ordinal_not_brier():
    frame=pd.DataFrame({'answered':[1],'predicted_error':[.2],'loss':[.25]})
    with pytest.raises(ValueError):t.calibrate(frame)

def test_perfect_calibration():
    frame=pd.DataFrame({'answered':[1]*4,'predicted_error':[0.,0.,1.,1.],'loss':[0.,0.,1.,1.]})
    assert t.calibrate(frame)==(0.,0.)

def test_constant_sum_is_not_coverage_probability():
    # Two perfectly correlated Bernoulli(.4)-correct agents: sum marginals=.8,
    # yet the chance that either is correct is still .4, not .8.
    probability=np.array([.6,.4]);correct=np.array([[0,0],[1,1]])
    assert (probability@correct).sum()==pytest.approx(.8)
    assert probability@correct.max(axis=1)==pytest.approx(.4)

def test_typed_does_not_mean_correct():
    obs={0:'wrong'};actions=d.legal_actions(obs)
    probs={k:float(k=='emit_0') for k in actions}
    result=d.validate_response({'answers':{'next_action':{'type':'choice','choice':'emit_0','probabilities':probs,'confidence':1.}}},obs)
    assert result.choice=='emit_0' and obs[0]!='correct'

@pytest.mark.parametrize('obs',[{}, {0:'A'},{0:'A',2:None},{0:'A',1:'B',3:'C'}])
def test_typed_actions_authorized(obs):
    actions=d.legal_actions(obs)
    for name in actions:
        if name.startswith('query_'):
            indices=list(map(int,name.split('_')[1:]))
            assert not set(indices)&set(obs) and len(indices)+len(obs)<=3
        if name.startswith('emit_'):assert int(name.split('_')[1]) in obs
    assert len(actions)<=255

@pytest.mark.parametrize('kind',['unknown','missing','nan','sum','confidence','notmaximum'])
def test_invalid_typed_decisions_rejected(kind):
    obs={0:'A'};a=d.legal_actions(obs);p={k:float(k=='emit_0') for k in a}
    answer={'type':'choice','choice':'emit_0','probabilities':p,'confidence':.8}
    if kind=='unknown':answer['choice']='emit_3'
    if kind=='missing':p.pop('defer')
    if kind=='nan':p['emit_0']=float('nan')
    if kind=='sum':p['emit_0']=.8
    if kind=='confidence':answer['confidence']=2
    if kind=='notmaximum':answer['choice']='defer'
    with pytest.raises(ValueError):d.validate_response({'answers':{'next_action':answer}},obs)

def test_contract_has_no_gold_field():
    request=d.request_for_state({0:'A'},'query',model='recorded-model-version')
    assert set(request)=={'model','state','questions'}
    assert 'ground_truth' not in str(request)

def test_mmlu_numpy_repr_is_parsed_without_execution():
    text="{'question': 'which?', 'subject': 'logic', 'choices': array(['a','b','c','d'], dtype=object)}"
    # Known inert dtype symbol is accepted without resolving/calling it.
    assert t.question_text(text,'mmlu')=='which?\na\nb\nc\nd'
    text=text.replace('dtype=object',"dtype='<U1'")
    assert t.question_text(text,'mmlu')=='which?\na\nb\nc\nd'

def test_mmlu_code_injection_rejected():
    with pytest.raises(ValueError):t.question_text("{'question':'q','choices':__import__('os').system('echo injected')}",'mmlu')

def test_tolerance_is_secondary_not_primary():
    from audit_transfer import numeric_equal
    a=t.normalize_answer('4.0000000000000004','svamp');b=t.normalize_answer('4','svamp')
    assert a!=b and numeric_equal(a,b)
    assert not numeric_equal(t.normalize_answer('4.01','svamp'),b)

def test_missing_test_suite_rejected(tmp_path):
    from ci_transfer import check_xml
    path=tmp_path/'tests.xml';path.write_text('<testsuite tests="999"></testsuite>')
    with pytest.raises(AssertionError):check_xml(path)

def test_skipped_test_rejected(tmp_path):
    from ci_transfer import check_xml
    path=tmp_path/'tests.xml';path.write_text('<testsuite>'+'<testcase/>'*201+'<testcase><skipped/></testcase></testsuite>')
    with pytest.raises(AssertionError):check_xml(path)

@pytest.mark.parametrize('value,expected',[
    ('a pedestrian','pedestrian'),('Left','left'),('Yes','yes'),('the table','table'),
    ('Two people','2 people'),('UPS truck','ups truck'),('  ',None),('',None),('NaN',None),
])
def test_gqa_answer_identity(value,expected):
    assert t.normalize_answer(value,'gqa')==expected

@pytest.mark.parametrize('value,dataset,expected',[
    ('yes','jigsaw','yes'),('No','jigsaw','no'),('toxic','jigsaw','yes'),('non-toxic','jigsaw','no'),
    (':','jigsaw',None),('maybe','jigsaw',None),('Unknown','gqa','unknown'),
])
def test_added_dataset_answer_identity(value,dataset,expected):
    assert t.normalize_answer(value,dataset)==expected

def test_gqa_gold_uses_the_same_identity():
    assert t.normalize_gold('a pedestrian','gqa')=='pedestrian'
    assert t.normalize_gold('Yes','gqa')=='yes'

@pytest.mark.parametrize('value,expected',[
    ('0.0','no'),('0.5','no'),('0.5000001','yes'),('1.0','yes'),('nan',None),('abc',None),
])
def test_jigsaw_verdict_threshold_is_strict(value,expected):
    # The archive's own label column treats exactly 0.5 as non-toxic; verified
    # against all comparable pairs rather than assumed.
    assert t.normalize_gold(value,'jigsaw')==expected

@pytest.mark.parametrize('dataset',['gqa','jigsaw','gsm8k','mmlu','strategyqa','svamp','dices'])
def test_upstream_debug_artifact_is_never_an_answer(dataset):
    assert t.normalize_answer('City stre[DEBUG] idx=120 | pred=Horse',dataset) is None

def test_gqa_prompt_keeps_image_identity():
    one="{'question': 'The vehicle is in front of who?', 'image_id': 2374257}"
    two=one.replace('2374257','2374258')
    assert t.question_text(one,'gqa').endswith('image 2374257')
    assert t.group_id(t.question_text(one,'gqa'))!=t.group_id(t.question_text(two,'gqa'))

@pytest.mark.parametrize('bad',["{'question': 'q'}","{'image_id': 1}","__import__('os').system('echo injected')"])
def test_gqa_prompt_schema_and_injection_rejected(bad):
    with pytest.raises(ValueError):t.question_text(bad,'gqa')

def test_primary_datasets_match_locked_protocol():
    import json
    p=json.loads((t.ROOT/'transfer_protocol.json').read_text())
    assert tuple(p['primary_datasets'])==t.PRIMARY_DATASETS
    assert set(t.PRIMARY_DATASETS)<=set(p['datasets'])
    assert set(p['source_sha256'])==set(p['datasets'])
    assert all(re.fullmatch('[0-9a-f]{64}',h) for h in p['source_sha256'].values())
    assert p['ordinal_exploratory_dataset'] not in t.PRIMARY_DATASETS

def test_locked_protocol_digest_matches_ci_pin():
    pinned=re.search(r"if protocol!='([0-9a-f]{64})'",(t.ROOT/'ci_transfer.py').read_text()).group(1)
    assert pinned==t.digest(t.ROOT/'transfer_protocol.json')

def test_openjev_protocol_and_shards_match_the_widened_pilot():
    import json
    from openjev_sharded import SHARDS
    p=json.loads((t.ROOT/'openjev_protocol.json').read_text())
    assert tuple(p['datasets'])==t.PRIMARY_DATASETS
    assert p['per_dataset']=={'test_groups':32,'development_groups':16,'reverse_order_test_groups':8,'repeated_forward_test_groups':2}
    assert all(s>0 for s in SHARDS.values()) and set(SHARDS)=={'qwen35-4b','qwen35-0.8b'}

def test_transfer_protocol_keeps_the_published_split_salt():
    # Changing this string would silently reshuffle the four datasets whose
    # splits are already published, so it must stay byte-identical.
    import json
    p=json.loads((t.ROOT/'transfer_protocol.json').read_text())
    assert 'dtmoa-proeval-v13' in p['split']

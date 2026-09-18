import json
import numpy as np
import pytest
import openjev_benchmark as b

def case():
    return dict(id='owned:1',dataset='svamp',sample_id='1',group='a'*64,split='test',task='2+2?',panel=[0,1,3],display_answers={'0':'4','1':'5','3':'4'},estimated_errors={'0':.1,'1':.3,'3':.2},evaluation_errors=[0.,1.,1.,0.])

def test_gold_never_rendered():
    c=case();a=b.render_request(c);c['evaluation_errors']=[1.,0.,0.,1.];assert b.render_request(c)==a
    assert 'evaluation_errors' not in json.dumps(a)

def test_unacquired_answer_never_rendered():
    c=case();c['display_answers']['2']='TOP_SECRET_UNACQUIRED';assert 'TOP_SECRET' not in json.dumps(b.render_request(c))

def test_reverse_same_options_state():
    c=case();a=b.render_request(c);r=b.render_request(c,True)
    assert a['state']==r['state'] and a['options']==list(reversed(r['options']))

def test_candidate_proposal_no_gold():
    result={'option_ids':['emit_3','defer','emit_0','emit_1'],'probabilities':[.4,.5,.06,.04]}
    selected,x,q=b.candidate_proposal(result);assert selected==3 and q==.4 and len(x)==2

@pytest.mark.parametrize('bad',[[.5,.5,.5,.5],[float('nan'),0,0,1],[-1,1,0,1],[1,0,0]])
def test_invalid_scores_rejected(bad):
    request=b.render_request(case());r={'option_ids':[x['id'] for x in request['options']],'probabilities':bad}
    with pytest.raises(ValueError):b.validate_result(r,request)

def test_unauthorized_option_rejected():
    request=b.render_request(case());r={'option_ids':['emit_2','emit_0','emit_1','defer'],'probabilities':[1,0,0,0]}
    with pytest.raises(ValueError):b.validate_result(r,request)

@pytest.mark.parametrize('all_correct',[0,1])
def test_constant_calibration_smoothing(all_correct):
    c=b.SuccessCalibrator().fit([[0,.5]]*8,[all_correct]*8)
    p=c.predict([[1,.2]])[0];assert 0<p<1 and p==pytest.approx((8*all_correct+1)/10)

def test_calibration_deterministic():
    x=np.array([[i/10,.5] for i in range(20)]);y=np.array([0]*10+[1]*10)
    a=b.SuccessCalibrator().fit(x,y);c=b.SuccessCalibrator().fit(x,y)
    np.testing.assert_array_equal(a.predict(x),c.predict(x))

def test_task_mutation_changes_hash():
    c=case();a=b.hash_object(b.render_request(c));c['task']='3+3?';assert b.hash_object(b.render_request(c))!=a

def test_request_four_choices():
    c=case();ids=[o['id'] for o in b.render_request(c)['options']]
    assert set(ids)=={'emit_0','emit_1','emit_3','defer'}

def test_protocol_constants():
    assert b.TEST_N==32 and b.DEV_N==16 and b.PRICE==.01 and b.DEFER==.25 and len(b.MODELS)==2

def test_do_not_invent_jev_weights():
    assert all(name.startswith('Qwen/') for name,rev in b.MODELS.values())

def test_source_revision_immutable():
    import re
    assert re.fullmatch('[0-9a-f]{40}',b.UPSTREAM)
    assert all(re.fullmatch('[0-9a-f]{40}',rev) for _,rev in b.MODELS.values())

def test_widened_pilot_counts_are_derived_and_pinned():
    # Derived gates scale with the locked protocol; the literal expectation here is
    # deliberate, so widening the pilot again requires editing this test instead of
    # silently passing stale numbers.
    import analyze_openjev as a
    nds=len(b.DATASETS)
    assert nds==6 and nds==len(b.DATASETS)
    assert a.TEST_TOTAL==b.TEST_N*nds==192 and a.DEV_TOTAL==b.DEV_N*nds==96
    assert a.CASE_TOTAL==288 and a.ROBUST_TOTAL==(b.ORDER_N+b.REPEAT_N)*nds==60
    from openjev_sharded import SHARDS
    assert SHARDS=={'qwen35-4b':12,'qwen35-0.8b':3}

def test_no_stale_hardcoded_pilot_size():
    # Regression guard for the v1.5 de-hardcoding: the shard and analysis gates must
    # derive their case counts from the locked protocol, so the old 128/64/32/40
    # literals must not reappear in them or in their comments.
    import re
    for name in ('analyze_openjev.py','openjev_sharded.py'):
        source=(b.ROOT/name).read_text()
        found=sorted(set(re.findall(r'(?<![\w.])(?:128|192|64|32|40)(?![\w.])',source)))
        assert not found,(name,found)

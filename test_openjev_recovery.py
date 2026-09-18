import copy
import numpy as np
import pytest
import openjev_benchmark as b
from recover_openjev_analysis import recover_request

def request():
    return {'id':'owned','state':{'task':'owned test, no benchmark gold','candidates':[{'candidate':i,'answer':str(i),'estimated_error_from_calibration':v} for i,v in enumerate((.1210212854882527,.12100337343642713,.9071275039555781))]},'question':'select','options':[{'id':str(i)} for i in range(3)]}

def test_exact_request_retained():
    r=request();got,offset=recover_request(r,b.hash_object(r));assert got==r and offset==[0,0,0]

def test_floating_request_recovered_exactly():
    r=request();target=copy.deepcopy(r);target['state']['candidates'][0]['estimated_error_from_calibration']=float(np.nextafter(r['state']['candidates'][0]['estimated_error_from_calibration'],-np.inf))
    got,offset=recover_request(r,b.hash_object(target));assert got==target and offset==[-1,0,0]

def test_changed_task_cannot_be_recovered():
    r=request();target=copy.deepcopy(r);target['state']['task']='different task'
    with pytest.raises(AssertionError):recover_request(r,b.hash_object(target),bound=1)

def test_changed_candidate_cannot_be_recovered():
    r=request();target=copy.deepcopy(r);target['state']['candidates'][0]['answer']='different answer'
    with pytest.raises(AssertionError):recover_request(r,b.hash_object(target),bound=1)

def test_substantive_risk_drift_rejected():
    r=request();target=copy.deepcopy(r);target['state']['candidates'][0]['estimated_error_from_calibration']+=1e-4
    with pytest.raises(AssertionError):recover_request(r,b.hash_object(target))

"""A Jev-compatible *request/response contract*, not a Jev model implementation.

No API call is made. An authenticated live adapter must separately record model
version, full allowed-action state, calibrated correctness and end-to-end cost.
Choice probabilities over actions are NOT action-success probabilities.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import combinations

@dataclass(frozen=True)
class TypedDecision:
    choice: str
    probabilities: dict[str,float]
    confidence: float

def legal_actions(observed:dict[int,str|None],budget:int=3,models:int=4):
    if not 0<=budget<=models or len(observed)>budget or any(type(i) is not int or i not in range(models) for i in observed):
        raise ValueError('Invalid authorization state')
    actions={'defer':'Stop and defer; do not invent an answer.'}
    for i in sorted(observed):actions[f'emit_{i}']=f'Return already acquired candidate {i}.'
    remaining=[i for i in range(models) if i not in observed]
    for k in range(1,budget-len(observed)+1):
        for batch in combinations(remaining,k):
            actions['query_'+'_'.join(map(str,batch))]='Acquire only candidate(s) '+','.join(map(str,batch))+'.'
    return actions

def request_for_state(observed,context,budget=3,model='jev-latest'):
    criteria=legal_actions(observed,budget)
    return {'model':model,'state':{'context':context,'observed':observed,'remaining_budget':budget-len(observed)},
        'questions':{'next_action':{'type':'choice','instructions':'Select an authorized next information-acquisition or terminal action. Respect the supplied loss and evidence; choice confidence is not a correctness guarantee.','criteria':criteria}}}

def validate_response(response,observed,budget=3):
    permitted=legal_actions(observed,budget)
    try:
        answer=response['answers']['next_action']
        if answer['type']!='choice':raise ValueError('Wrong primitive')
        probs=answer['probabilities'];choice=answer['choice'];confidence=float(answer['confidence'])
        if set(probs)!=set(permitted) or choice not in permitted:raise ValueError('Action-set mismatch')
        values={k:float(v) for k,v in probs.items()}
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values.values()) or abs(sum(values.values())-1)>1e-6:raise ValueError('Invalid probability simplex')
        if not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid confidence')
        if values[choice]<max(values.values())-1e-6:raise ValueError('Choice is not a maximum-probability option')
        return TypedDecision(choice,values,confidence)
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        raise ValueError('Untrusted decision response rejected') from exc

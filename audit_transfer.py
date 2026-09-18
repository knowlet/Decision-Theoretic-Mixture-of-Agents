"""Post-primary data-quality diagnostics. Never change primary policy labels."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import transfer_study as t

def numeric_equal(prediction,gold,rtol=1e-9,atol=1e-12):
    if prediction is None or gold is None:return False
    from fractions import Fraction
    try:
        a=float(Fraction(prediction));b=float(Fraction(gold))
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol)
    except (ValueError,ZeroDivisionError,OverflowError):return False

def audit(cache,out):
    p=json.loads((t.ROOT/'transfer_protocol.json').read_text());rows=[];differences=[];bounds=[]
    for dataset in p['datasets']:
        path=cache/f'{dataset}_predictions.csv'
        if t.digest(path)!=p['source_sha256'][dataset]:raise ValueError('Source hash changed')
        d=pd.read_csv(path,dtype=str,keep_default_na=False)
        for pool,models in t.POOL_NAMES.items():
            for model in models:
                labels=pd.to_numeric(d['label_'+model],errors='coerce')
                bounds.append(dict(dataset=dataset,model=model,n=len(d),missing=int(labels.isna().sum()),minimum=labels.min(),maximum=labels.max(),outside_documented_0_1=int(((labels<0)|(labels>1)).sum())))
                if dataset not in ('gsm8k','svamp'):continue
                exact_diff=0;tol_diff=0;rounding=0
                for r in d.to_dict('records'):
                    answer=t.normalize_answer(r['prediction_'+model],dataset);gold=t.normalize_gold(r['ground_truth'],dataset)
                    label=float(r['label_'+model]);exact=float(answer is None or answer!=gold);tol=float(not numeric_equal(answer,gold))
                    exact_diff+=int(exact!=label);tol_diff+=int(tol!=label);rounding+=int(exact!=tol)
                    if exact!=label or tol!=label:
                        differences.append(dict(dataset=dataset,model=model,sample_id=r['index'],exact_error=exact,tolerance_error=tol,upstream_error=label,normalization_changed=int(exact!=tol)))
                rows.append(dict(dataset=dataset,pool=pool,model=model,n=len(d),exact_disagreements=exact_diff,tolerant_disagreements=tol_diff,rounding_only_changes=rounding))
    pd.DataFrame(rows).to_csv(out/'numeric_tolerance_audit.csv',index=False)
    pd.DataFrame(differences).to_csv(out/'numeric_disagreements.csv',index=False)
    pd.DataFrame(bounds).to_csv(out/'label_range_audit.csv',index=False)
    t.write_json(out/'audit_scope.json',{'status':'Post-primary diagnostic; no policy changes, refitting, exclusions, or primary-label changes',
        'numeric_tolerance':{'rtol':1e-9,'atol':1e-12},'normalization':'Primary uses exact rational normalization; this diagnostic tests machine-precision aliases against the same archived gold, not an independent original-benchmark source.',
        'dices':'Values above the documented [0,1] range are flagged, not automatically repaired. The prespecified complete-case subset is exploratory only; all-deferral is not evidence of adequate human preference decisions.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=t.ROOT/'.cache/proeval');p.add_argument('--out',type=Path,default=t.ROOT/'results/transfer');a=p.parse_args();audit(a.cache,a.out)

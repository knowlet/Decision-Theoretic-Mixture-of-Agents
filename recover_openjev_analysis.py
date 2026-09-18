"""Analysis-only recovery of immutable successful inference shards.
Exact request SHA-256 remains mandatory. Only a bounded +/-8 IEEE-754-ULP
neighborhood of three TRAINING-derived errors is searched. All other fields
remain identical. No gold or inference output is optimized. Source inference
and later recovery/reporting provenance remain separate.
"""
from __future__ import annotations
import argparse,copy,itertools,json,os
from pathlib import Path
import numpy as np
import openjev_benchmark as b
import openjev_sharded as sh
import analyze_openjev as az
import transfer_study as t
SOURCE_COMMIT='ee971cc25e1475ac16375886c135fe155a0dc803'
SOURCE_RUN='35310006824'
MAX_ULPS=8

def nearby(value,bound=MAX_ULPS):
    values=[(float(value),0)];lo=hi=float(value)
    for step in range(1,bound+1):
        lo=float(np.nextafter(lo,-np.inf));hi=float(np.nextafter(hi,np.inf))
        values.extend(((lo,-step),(hi,step)))
    return values

def recover_request(request,target_hash,bound=MAX_ULPS):
    result=copy.deepcopy(request);candidates=result['state']['candidates']
    if len(candidates)!=3:raise ValueError('Expected three authorized candidates')
    base=[float(c['estimated_error_from_calibration']) for c in candidates]
    if b.hash_object(result)==target_hash:return result,[0,0,0]
    for combination in itertools.product(*(nearby(v,bound) for v in base)):
        values,ulps=zip(*combination)
        if any(not np.isfinite(v) or not 0<=v<=1 for v in values):continue
        for c,value in zip(candidates,values):c['estimated_error_from_calibration']=value
        if b.hash_object(result)==target_hash:
            if max(abs(x-y) for x,y in zip(base,values))>1e-12:raise AssertionError('Semantic numeric drift')
            return result,list(ulps)
    raise AssertionError('Exact source request was NOT recovered in the fixed numeric neighborhood')

def restore(label,incoming,cache,out):
    original_render=b.render_request;cases,*_=b.prepare(cache);case_map={c['id']:c for c in cases}
    reconstructed={};audits=[]
    for index in range(sh.SHARDS[label]):
        paths=list(incoming.glob(f'openjev-shard-{label}-{index}-*/results/shard'))
        if len(paths)!=1:raise AssertionError('Missing/duplicate successful shard')
        folder=paths[0]
        for name,sha in json.loads((folder/'SHA256.json').read_text()).items():
            if Path(name).name!=name or t.digest(folder/name)!=sha:raise AssertionError('Source artifact changed')
        meta=json.loads((folder/'provenance.json').read_text())
        if meta['tested_commit']!=SOURCE_COMMIT:raise AssertionError('Wrong inference source')
        for line in (folder/'scores.jsonl').read_text().splitlines():
            record=json.loads(line);cid=record['case_id'];kind=record['kind'];reverse=kind=='reverse'
            if cid not in case_map or kind not in ('primary','repeat','reverse'):raise AssertionError('Unknown inference case')
            expected=original_render(case_map[cid],reverse)
            request,ulps=recover_request(expected,record['request_sha256'])
            key=(cid,reverse)
            if key in reconstructed and request!=reconstructed[key]:raise AssertionError('Repeat request differs')
            reconstructed[key]=request;b.validate_result(record['result'],request)
            old=[x['estimated_error_from_calibration'] for x in expected['state']['candidates']]
            new=[x['estimated_error_from_calibration'] for x in request['state']['candidates']]
            audits.append({'model':label,'shard':index,'case_id':cid,'kind':kind,'exact_request_sha256':record['request_sha256'],
                'reference_request_sha256':b.hash_object(expected),'ulp_offsets':ulps,'max_absolute_difference':max(abs(x-y) for x,y in zip(old,new)),
                'source_estimates':new,'reference_estimates':old,'all_other_request_fields_unchanged':True})
    if len(audits)!=232 or len(reconstructed)!=224:raise AssertionError('Incomplete inference inventory')
    def exact_render(case,reverse=False):return copy.deepcopy(reconstructed[(case['id'],reverse)])
    previous=os.environ.get('GITHUB_SHA');os.environ['GITHUB_SHA']=SOURCE_COMMIT;b.render_request=exact_render
    try:sh.finalize(label,incoming,cache,out)
    finally:
        b.render_request=original_render
        if previous is None:os.environ.pop('GITHUB_SHA',None)
        else:os.environ['GITHUB_SHA']=previous
    t.write_json(out/'request_numeric_recovery.json',audits)
    meta=json.loads((out/'provenance.json').read_text())
    meta.update(inference_run_url=f'https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/{SOURCE_RUN}',
        analysis_commit=os.getenv('ANALYSIS_COMMIT'),analysis_run_url=os.getenv('RESEARCH_RUN_URL'),
        request_reconstruction='All recorded SHA-256 values recovered exactly; only +/-8 ULP candidate errors searched',
        strict_byte_identical_requests_across_hosts=False)
    t.write_json(out/'provenance.json',meta)
    t.write_json(out/'SHA256.json',{p.name:t.digest(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256.json'})
    return audits

def run(incoming,cache,inputs,out):
    inputs.mkdir(parents=True,exist_ok=True);all_audits=[]
    for label in b.MODELS:all_audits.extend(restore(label,incoming,cache,inputs/label))
    original_validate=az.validate_bundle;manifests={}
    def validate_with_explicit_numeric_equivalence(path):
        frame,metadata,manifest=original_validate(path)
        records=json.loads((path/'request_numeric_recovery.json').read_text())
        assert len(records)==232 and all(r['all_other_request_fields_unchanged'] and max(map(abs,r['ulp_offsets']))<=MAX_ULPS for r in records)
        assert all(r['max_absolute_difference']<=1e-12 for r in records)
        manifests[metadata['selector_model']]=manifest
        # Strict identity/panel/split/group match. Literal decimal request hashes
        # can differ across CPUs and were reconstructed independently above.
        core=[{k:v for k,v in r.items() if k!='request_sha256'} for r in manifest]
        return frame,metadata,core
    previous=os.environ.get('GITHUB_SHA');os.environ['GITHUB_SHA']=SOURCE_COMMIT;az.validate_bundle=validate_with_explicit_numeric_equivalence
    try:az.run(inputs,out)
    finally:
        az.validate_bundle=original_validate
        if previous is None:os.environ.pop('GITHUB_SHA',None)
        else:os.environ['GITHUB_SHA']=previous
    t.write_json(out/'request_numeric_recovery.json',all_audits)
    mismatched=sum(a['exact_request_sha256']!=a['reference_request_sha256'] for a in all_audits)
    cross=sum(x['request_sha256']!=y['request_sha256'] for x,y in zip(manifests['qwen35-4b'],manifests['qwen35-0.8b']))
    cross_test=sum(x['split']=='test' and x['request_sha256']!=y['request_sha256'] for x,y in zip(manifests['qwen35-4b'],manifests['qwen35-0.8b']))
    audit={'all_request_hashes_matched_exactly':True,'records':len(all_audits),'records_requiring_numeric_recovery':mismatched,
        'max_absolute_difference':max(a['max_absolute_difference'] for a in all_audits),
        'max_ulp_offset':max(max(map(abs,a['ulp_offsets'])) for a in all_audits),
        'primary_request_hash_differences_between_models':cross,'test_request_hash_differences_between_models':cross_test,
        'same_task_panel_split_group':True,'same_literal_numeric_input_between_models':cross==0,
        'scope':'Post-inference serialization recovery, not a sensitivity experiment. Tokenization can differ for last-bit decimal representations; no bit-identical cross-host input claim.'}
    t.write_json(out/'numeric_recovery_summary.json',audit)
    meta=json.loads((out/'verification.json').read_text())
    meta.update(inference_commit=SOURCE_COMMIT,inference_run_url=f'https://github.com/knowlet/Decision-Theoretic-Mixture-of-Agents/actions/runs/{SOURCE_RUN}',
        analysis_commit=os.getenv('ANALYSIS_COMMIT'),analysis_run_url=os.getenv('RESEARCH_RUN_URL'),
        original_aggregate_job_conclusion='failure at cross-host request-hash equality; all ten inference jobs succeeded',
        request_recovery=audit,analysis_only=True)
    t.write_json(out/'verification.json',meta);print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--incoming',type=Path,default=Path('incoming'));p.add_argument('--cache',type=Path,default=Path('.cache/proeval'));p.add_argument('--inputs',type=Path,default=Path('inputs'));p.add_argument('--out',type=Path,default=Path('results/openjev'));p.add_argument('--download',action='store_true');a=p.parse_args()
    if a.download:t.download(a.cache)
    run(a.incoming,a.cache,a.inputs,a.out)

"""Generate both v1.3 manuscripts from verified result tables, not hand-filled scores."""
import json,shutil,re
from pathlib import Path
import pandas as pd
import transfer_study as t

R=Path('results/transfer');OUT=Path('dist-transfer')

def table(frame,columns,labels=None,digits=6):
    labels=labels or columns
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for _,r in frame.iterrows():
        v=[]
        for c in columns:
            x=r[c]
            if pd.isna(x):s='—'
            elif isinstance(x,float):s=f'{x:.{digits}f}'
            else:s=str(x)
            s=s.replace('replacement_refit','R').replace('replacement_frozen','F').replace('legacy_id','L')
            v.append(s.replace('|','/'))
        lines.append('| '+' | '.join(v)+' |')
    return '\n'.join(lines)

def build():
    OUT.mkdir(exist_ok=True)
    evidence=json.loads(Path('verification/transfer.json').read_text())
    if not evidence['all_gates_passed']:raise RuntimeError('Unverified manuscript prohibited')
    s=pd.read_csv(R/'summary.csv');c=pd.read_csv(R/'paired_comparisons.csv');a=json.loads((R/'source_audit.json').read_text())
    b=s[s.dataset.isin(t.PRIMARY_DATASETS)]
    macro=b.groupby(['phase','policy'],sort=True).agg(objective=('objective','mean'),coverage=('coverage','mean'),queries=('queries','mean')).reset_index()
    core=macro[macro.policy.isin(['single','static','myopic','bellman','prompt_top1','cumulative_score']) & macro.phase.isin(['legacy_id','replacement_frozen','replacement_refit'])]
    dataset_rows=pd.DataFrame([dict(dataset=x['dataset'],source=x['source_rows'],retained=x['retained_rows'],groups=x['unique_groups'],train=x['split_counts']['train'],dev=x['split_counts']['dev'],test=x['split_counts']['test']) for x in a])
    bytask=s[(s.phase=='replacement_refit')&s.policy.isin(['single','static','myopic','bellman'])]
    paired=c[c.dataset=='macro_binary']
    refresh=pd.read_csv(R/'refresh_costs.csv');refresh=refresh[refresh.policy=='bellman']
    calibration=b[(b.phase.isin(['replacement_frozen','replacement_refit']))&(b.policy=='bellman')]
    gain=paired[paired.a=='replacement_frozen:bellman'].iloc[0]
    refit_objective=macro[(macro.phase=='replacement_refit')&(macro.policy=='bellman')].objective.iloc[0]
    replacements={
       'REFIT_GAIN':f"{gain['difference']:.6f}",'REFIT_GAIN_LO':f"{gain['lo']:.6f}",'REFIT_GAIN_HI':f"{gain['hi']:.6f}",'REFIT_OBJECTIVE':f'{refit_objective:.6f}',
       'DATASETS':table(dataset_rows,['dataset','source','retained','groups','train','dev','test'],digits=0),
       'MACRO':table(core,['phase','policy','objective','coverage','queries']),
       'TASKS':table(bytask,['dataset','policy','n','objective','coverage','queries']),
       'PAIRS':table(paired,['a','b','difference','lo','hi']),
       'REFRESH':table(refresh,['dataset','budget','actual_calibration_questions','assumed_refresh_cost','per_query_gain','break_even_future_queries'],['dataset','budget','n_cal','K','gain','N_break']),
       'CALIBRATION':table(calibration,['dataset','phase','brier','ece']),
       'TESTS':str(evidence['tests']),'COMMIT':evidence['tested_commit'],
       'RUN_URL':evidence.get('run_url') or 'local build (not hosted execution)',
       'OUTPUTS':str(len(evidence['comparisons'])),
       'IDENTICAL':str(sum(x['byte_identical'] for x in evidence['comparisons'])),
    }
    for lang in ('en','zh-TW'):
        text=Path(f'transfer_paper.{lang}.md').read_text()
        for key,val in replacements.items():text=text.replace('@@'+key+'@@',val)
        if '@@' in text:raise RuntimeError('Unfilled manuscript value')
        text=re.sub(r'https?://[^\s<>。]+',lambda m:'<'+m.group().rstrip('.,')+'>'+m.group()[len(m.group().rstrip('.,')):],text)
        (OUT/f'paper.{lang}.md').write_text(text)
    for name in ('summary.csv','paired_comparisons.csv','refresh_costs.csv','source_audit.json','numeric_tolerance_audit.csv','label_range_audit.csv'):
        shutil.copy(R/name,OUT/name)
    shutil.copy('verification/transfer.json',OUT/'verification.json')

if __name__=='__main__':build()

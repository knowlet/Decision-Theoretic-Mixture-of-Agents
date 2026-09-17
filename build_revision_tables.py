"""Tables are generated from executed CSVs; no fabricated benchmark entries."""
from pathlib import Path
import pandas as pd

def table(head, rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def replacements(root: Path):
    r=root/'results'/'revision'; m=pd.read_csv(r/'matched_summary.csv')
    learned=pd.read_csv(r/'calibration_curve.csv').groupby(['n_per_type','policy']).objective.mean().unstack()
    route=pd.read_csv(r/'routing_noise.csv').groupby(['route_error','policy']).objective.mean().unstack()
    runtime=pd.read_csv(r/'planning_runtime.csv').groupby('policy').five_type_plan_seconds.median()
    matched=table(['策略 / Policy','$J$ [95% CI]','Cost','Latency'],[
        [x.policy,f'{x.objective:.4f} [{x.objective_lo:.4f}, {x.objective_hi:.4f}]',f'{x.cost:.3f}',f'{x.latency:.3f}']
        for x in m.itertuples()])
    learning=table(['n / type','C3','P1','P1 independent'],[
        [n]+[f'{learned.loc[n,name]:.4f}' for name in ['C3_greedy','P1_bellman','P1_independence']] for n in learned.index])
    routing=table(['Error rate','C3','P0','P1','R1'],[
        [f'{eta:.0%}']+[f'{route.loc[eta,name]:.4f}' for name in ['C3_greedy','P0_heuristic','P1_bellman','R1_static_all']] for eta in route.index])
    timing=table(['Policy','Median seconds / five-type plan'],[[name,f'{value:.5f}'] for name,value in runtime.items()])
    sub={'MATCHED_ZH':matched,'LEARNING_ZH':learning,'ROUTING_ZH':routing,'RUNTIME_ZH':timing}
    section=(root/'revision_section.zh.md').read_text()
    for k,v in sub.items(): section=section.replace('@@'+k+'@@',v)
    sub['REVISION_SECTION']=section
    return sub

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    repl=replacements(root)
    src=(root/'paper.en.template.md').read_text()
    for k,v in repl.items():src=src.replace('@@'+k+'@@',v)
    assert '@@' not in src
    (root/'paper.en.md').write_text(src)

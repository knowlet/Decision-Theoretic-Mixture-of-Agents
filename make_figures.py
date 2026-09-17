"""Publication figures, generated only from saved experiment outputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
R=Path(__file__).resolve().parent/'results'
F=R.parent/'figures';F.mkdir(exist_ok=True)
S=pd.read_csv(R/'summary.csv').set_index('policy')
D=pd.read_csv(R/'metrics_by_replicate.csv').pivot(index='replicate',columns='policy',values='objective')
main=['C1_single','C2_fixed3','C3_greedy','P0_heuristic','P1_bellman']
labels=['C1: single worker','C2: fixed three-worker fusion','C3: myopic cascade','P0: proposed heuristic','P1: fitted Bellman policy']
# Each figure uses a separate axes; Matplotlib's default color cycle is unchanged.
fig,ax=plt.subplots(figsize=(8.2,4.2))
y=np.arange(len(main)); vals=S.loc[main,'objective'].to_numpy(); lo=S.loc[main,'objective_lo'].to_numpy();hi=S.loc[main,'objective_hi'].to_numpy()
ax.barh(y,vals,xerr=np.array([vals-lo,hi-vals]),capsize=3)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Mean decision loss + 0.1 cost + 0.04 latency (lower is better)')
ax.set_xlim(0,1.2)
for i,v in enumerate(vals):ax.text(v+.02,i,f'{v:.4f}',va='center',fontsize=10)
ax.set_title('Primary simulation: 20 replicates, 400,000 held-out cases')
fig.tight_layout();fig.savefig(F/'primary_objective.png',dpi=210);plt.close(fig)

abl=['P1_no_verifier','P1_no_elicitor','P1_no_diverse','P1_independence','P1_force_budget','C3_greedy_batch']
alab=['Remove verifier V','Remove preference elicitation E','Remove diverse workers C/D','Assume conditional independence','Force all four acquisitions','Myopic batch look-ahead']
means=[];ci=[]
for a in abl:
 d=D[a]-D['P1_bellman'];means.append(d.mean());ci.append(stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
fig,ax=plt.subplots(figsize=(8.2,4.2));yy=np.arange(len(abl))
ax.errorbar(means,yy,xerr=ci,fmt='o',capsize=4)
ax.axvline(0,linestyle='--',linewidth=1)
ax.set_yticks(yy,alab);ax.invert_yaxis();ax.set_xlabel('Paired change in objective relative to P1 (positive = worse)')
ax.set_title('Ablations: not every proposed component is necessary')
fig.tight_layout();fig.savefig(F/'ablation_effects.png',dpi=210);plt.close(fig)

stress=pd.read_csv(R/'stress_exact.csv').groupby(['scenario','policy']).objective.mean().unstack()
scens=['common_failure_shift','verifier_degradation','elicitor_degradation','cost_shift','easy_only_shift']
short=['Shared errors','Bad verifier','Noisy preferences','Cost shock','Easy-only tasks']
fig,ax=plt.subplots(figsize=(8.4,4.8))
for a,label in zip(main,labels):ax.plot(range(5),stress.loc[scens,a],marker='o',label=label)
ax.set_xticks(range(5),short,rotation=15);ax.set_ylabel('Exact expected objective under shifted environment')
ax.set_title('Frozen policies: ranking reverses under distribution and cost shifts')
ax.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.01));ax.set_ylim(0,1.7)
fig.tight_layout();fig.savefig(F/'stress_shifts.png',dpi=210);plt.close(fig)

sw=pd.read_csv(R/'cost_latency_sweep.csv'); sw=sw[np.isclose(sw.mu,.04)]
fig,ax=plt.subplots(figsize=(7.8,4.7))
for a,lab in [('P1_oracle','Bellman (known distribution)'),('C3_oracle','Single-step greedy'),('C3_batch_oracle','Batch-greedy')]:
 d=sw[sw.policy==a].sort_values('cost')
 ax.plot(d.cost,d.terminal_loss,marker='o',label=lab)
ax.set_xlabel('Expected normalized acquisition cost');ax.set_ylabel('Expected terminal decision loss')
ax.set_title('Cost-quality trade-off (latency penalty fixed at 0.04)')
ax.legend(fontsize=9);fig.tight_layout();fig.savefig(F/'cost_quality.png',dpi=210);plt.close(fig)

# Extra: verifier dependency changes the comparison even with perfect calibration.
a=pd.read_csv(R/'exploratory_verifier_dependency.csv')
fig,ax=plt.subplots(figsize=(7.7,4.2))
for policy,lab in [('C3_greedy','Myopic cascade'),('P0_heuristic','P0 heuristic'),('P1_bellman','Bellman oracle')]:
 d=a[a.policy==policy].sort_values('candidate_required');ax.plot([0,1],d.objective,marker='o',label=lab)
ax.set_xticks([0,1],['V can directly answer','V requires earlier candidate'])
ax.set_ylabel('Exact expected objective');ax.set_title('Exploratory check: tool prerequisites change policy ranking')
ax.legend();fig.tight_layout();fig.savefig(F/'verifier_dependency.png',dpi=210);plt.close(fig)
print('Created',len(list(F.glob('*.png'))),'figures')

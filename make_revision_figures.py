from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
frame=pd.read_csv(root/'results/revision/calibration_curve.csv')
fig,ax=plt.subplots(figsize=(7.1,4.3))
for name,label in [('C3_greedy','C3: myopic'),('P1_bellman','P1: full joint'),('P1_independence','P1: independent approximation')]:
    part=frame[frame.policy==name].groupby('n_per_type').objective
    mean=part.mean(); half=part.std(ddof=1)/np.sqrt(part.count())*stats.t.ppf(.975,part.count()-1)
    ax.plot(mean.index,mean.values,marker='o',label=label)
    ax.fill_between(mean.index,(mean-half).values,(mean+half).values,alpha=.15)
ax.set_xscale('log');ax.set_xlabel('Calibration draws per task type');ax.set_ylabel('Exact expected composite loss (lower is better)')
ax.legend(fontsize=8);fig.tight_layout()
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures/calibration_curve.png',dpi=180)
plt.close(fig)

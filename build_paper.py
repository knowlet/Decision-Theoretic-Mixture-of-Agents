"""Assemble paper tables from saved results. No manual benchmark numbers."""
from pathlib import Path
import json, platform, importlib.metadata
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parent; R=ROOT/'results'
S=pd.read_csv(R/'summary.csv').set_index('policy')
D=pd.read_csv(R/'metrics_by_replicate.csv').pivot(index='replicate',columns='policy',values='objective')
META=json.loads((R/'run_metadata.json').read_text())
main=['C1_single','C2_fixed3','C3_greedy','P0_heuristic','P1_bellman']
short=dict(zip(main,['C1 單模型','C2 固定三模型','C3 單步級聯','P0 原啟發式','P1 Bellman']))
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
repl={}
repl['POLICY_TABLE']=table(['組別','資訊取得策略','重要限制'],[
 ['C1','依任務類型校準選單一工作模型','每題固定查一次 A–D 中的模型'],
 ['C2','依任務類型校準選固定三模型 panel','完整聯合後驗融合，不是多數決'],
 ['C3','單步 VOI，每次選一個通道','每步假設查完就停止，再滾動重算'],
 ['P0','風險分流 → 異質 panel → 分歧驗證／偏好澄清','本文對原文字提案的具體化'],
 ['P1','剩餘預算內的完整 Bellman 前瞻','可單查、平行、續查、停止或轉人工']])
repl['TASK_TABLE']=table(['任務類型','混合比例','錯選損失 $w_t$','主要不確定性'],[
 ['簡單事實', '30%','1.0','低共同失誤'],['困難事實','25%','5.0','較高共同失誤'],
 ['高風險事實','15%','20.0','相同困難度，後果更大'],['低利害偏好','15%','0.05','私人偏好未知，錯選代價小'],['高利害偏好','15%','8.0','私人偏好未知，錯選代價大']])
repl['GENERATOR_TABLE']=table(['設定','$P(G=1)$','$P(F=1)$','$r_A$','$r_B$','$r_C$','$r_D$'],[
 ['簡單事實','.015','.035','.95','.92','.90','.86'],['困難／高風險事實','.12','.08','.82','.80','.77','.74']])
repl['RESOURCE_TABLE']=table(['通道','功能','成本 $c_i$','延遲 $\\tau_i$'],[
 ['A','工作模型；與 B 有同家族失誤','1.00','1.00'],['B','同家族工作模型','0.80','0.90'],
 ['C','異質工作模型','0.80','0.80'],['D','較便宜的異質工作模型','0.55','0.65'],
 ['V','事實查證／grounded-answer','1.30','1.10'],['E','偏好澄清','0.65','2.00']])
rows=[]
for a in main:
 x=S.loc[a]
 rows.append([short[a],f'{x.objective:.4f} [{x.objective_lo:.4f}, {x.objective_hi:.4f}]',f'{x.terminal_loss:.4f}',f'{x.cost:.3f}',f'{x.latency:.3f}',f'{100*x.coverage:.2f}%'])
repl['MAIN_TABLE']=table(['策略','$J$ [95% CI]','終端損失','成本','延遲','Coverage'],rows)
p=pd.read_csv(R/'paired_comparisons.csv')
repl['PAIR_TABLE']=table(['比較','配對差','95% CI','Holm 校正 $p$'],[
 [short[r.a]+' − '+short[r.b],f'{r.difference:.5f}',f'[{r.lo:.5f}, {r.hi:.5f}]',f'{r.holm_p:.2e}'] for r in p.itertuples()])
c=pd.read_csv(R/'metrics_by_type.csv').groupby(['type','policy']).objective.mean().unstack()
types=['factual_easy','factual_hard','factual_critical','preference_low','preference_high']
zh=['簡單事實','困難事實','高風險事實','低利害偏好','高利害偏好']
repl['CATEGORY_TABLE']=table(['任務','C1','C2','C3','P0','P1'],[[z]+[f'{c.loc[t,a]:.4f}' for a in main] for t,z in zip(types,zh)])
abl=[('P1_no_verifier','去掉 V'),('P1_no_elicitor','去掉 E'),('P1_no_diverse','去掉 C/D'),('P1_independence','假設條件獨立'),('P1_force_budget','強制用完預算'),('C3_greedy_batch','單步批次前瞻')]
rows=[]
for a,label in abl:
 d=D[a]-D['P1_bellman'];m=d.mean();delta=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
 rows.append([label,f'{S.loc[a,"objective"]:.4f}',f'{m:+.5f}',f'[{m-delta:.5f}, {m+delta:.5f}]'])
repl['ABLATION_TABLE']=table(['變體','$J$','相對 P1 差值','差值 95% CI'],rows)
st=pd.read_csv(R/'stress_exact.csv').groupby(['scenario','policy']).objective.mean().unstack()
sc=['common_failure_shift','verifier_degradation','elicitor_degradation','cost_shift','easy_only_shift'];sz=['共同失誤增加','驗證器退化','偏好訊號退化','成本衝擊','僅剩簡單任務']
repl['STRESS_TABLE']=table(['漂移條件','C1','C2','C3','P0','P1'],[[z]+[f'{st.loc[t,a]:.4f}' for a in main] for t,z in zip(sc,sz)])
dep=pd.read_csv(R/'exploratory_verifier_dependency.csv').pivot(index='candidate_required',columns='policy',values='objective')
repl['DEPENDENCY_TABLE']=table(['V 的可行性','C3','P0','P1'],[[label]+[f'{dep.loc[b,a]:.4f}' for a in ['C3_greedy','P0_heuristic','P1_bellman']] for b,label in [(False,'可直接回答'),(True,'先有候選才可驗證')]])
repl['ENV_TABLE']=table(['項目','值'],[
 ['Python',platform.python_version()],['NumPy / SciPy',np.__version__+' / '+importlib.metadata.version('scipy')],
 ['Pandas / Matplotlib',pd.__version__+' / '+importlib.metadata.version('matplotlib')],
 ['測試框架','pytest '+importlib.metadata.version('pytest')+'；86 項測試'],['LLM API 呼叫','0'],['GPU','未使用'],
 ['主校準抽樣 / 留出測試','500,000 / 400,000'],['獨立主重複','20'],['新增學習曲線校準抽樣','2,660,000']])
repl['FILES_TABLE']=table(['檔案','用途'],[
 ['`study.py`','主生成器、策略、Bellman、評分、漂移、掃描'],['`test_study.py` / `test_revision.py`','86 個測試；含獨立 tuple-history 求解器'],
 ['`supplement.py`','事後探索性工具依賴與更新分布參照'],['`protocol.json`','首次執行前保存的本地協議'],
 ['`results/raw/`','20 組留出案例與校準 sufficient statistics'],['`summary.csv`','主指標與信賴區間'],
 ['`paired_comparisons.csv`','四個主比較與 Holm 校正'],['`stress_exact.csv`','凍結策略的分布漂移結果'],
 ['`cost_latency_sweep.csv`','28 組資源權重 × 3 策略'],['`policy_snapshots.json`','第一組重複的政策與查詢 trace'],
 ['`run_metadata.json`','執行環境、案例數與主協議雜湊'],['`verification/attestation.json`','測試、重跑、環境與來源雜湊'],['`revision.py`','資訊匹配、學習曲線與路由漂移'],['`revision_protocol.json`','事後探索性修訂協議']])
repl['RUNTIME']=f'{META["elapsed_seconds"]:.2f}'
from build_revision_tables import replacements
repl.update(replacements(ROOT))
text=(ROOT/'paper_template.md').read_text()
for k,v in repl.items():text=text.replace('@@'+k+'@@',v)
assert '@@' not in text
# Normalize occasional simplified characters introduced during drafting.
for a,b in {'选择':'選擇','檢验':'檢驗','改变':'改變','影响':'影響','误差':'誤差','合计':'合計','说明':'說明','对 P0':'對 P0','简单':'簡單','日志':'日誌','比较':'比較','独立':'獨立','多样性':'多樣性','结果':'結果'}.items():text=text.replace(a,b)
(ROOT/'paper.zh-TW.md').write_text(text)
print('Wrote paper.zh-TW.md:',len(text),'characters')

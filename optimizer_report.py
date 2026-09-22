"""Generate bilingual, result-derived notes only after independent evidence gates."""
from pathlib import Path
import json
import shutil
import pandas as pd

R=Path('results/optimizer');D=Path('dist-optimizer')
LABELS={'legacy_bellman':'Legacy Bellman','compiled_legacy':'Compiled legacy','polarity_bellman':'Answer polarity',
        'context_bellman':'Context only','context_polarity':'Context + polarity','tuning_selected':'Tuning-selected challenger',
        'guarded_selected':'Guarded deployment'}

def table(frame,columns):
    text=['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for _,r in frame.iterrows():
        text.append('| '+' | '.join((f'{r[c]:.6f}' if isinstance(r[c],float) else str(r[c])) for c in columns)+' |')
    return '\n'.join(text)


def main():
    D.mkdir(exist_ok=True);v=json.load(open('verification/optimizer.json'))
    if not v['all_gates_passed']:raise RuntimeError('Unverified results')
    m=pd.read_csv(R/'macro.csv');g=pd.read_csv(R/'promotion_gates.csv');c=pd.read_csv(R/'comparisons.csv');s=pd.read_csv(R/'summary.csv');rt=pd.read_csv(R/'runtime.csv')
    rows=m[m.seed==1601].copy();rows.method=rows.method.map(LABELS)
    tasks=s[(s.seed==1601)&s.method.isin(['legacy_bellman','polarity_bellman','context_polarity'])].pivot(index='dataset',columns='method',values='objective').reset_index()
    paired=c[c.dataset=='macro_six'][['method','difference','lo','hi','family_lo','family_hi']]
    timing=rt.pivot(index='dataset',columns='implementation',values='median_ns').reset_index();timing['speedup']=timing.legacy/timing.compiled
    class_rows=pd.read_csv(R/'class_outcomes.csv')
    class_rows=class_rows[(class_rows.seed==1601)&(class_rows.dataset=='jigsaw')&class_rows.method.isin(['legacy_bellman','polarity_bellman','context_polarity'])]
    class_table=table(class_rows,['method','gold','n','correct','wrong','deferred'])
    def cell(method, gold, column):
        row = class_rows[(class_rows.method == method) & (class_rows.gold == gold)]
        assert len(row) == 1, f'missing class diagnostic for {method}/{gold}'
        return int(row.iloc[0][column])
    nontoxic_n = cell('legacy_bellman', 'no', 'n')
    toxic_n = cell('legacy_bellman', 'yes', 'n')
    new_toxic_correct = {m: cell(m, 'yes', 'correct') for m in ('polarity_bellman', 'context_polarity')}
    new_nontoxic_wrong = {m: cell(m, 'no', 'wrong') for m in ('polarity_bellman', 'context_polarity')}
    legacy_nontoxic_wrong = cell('legacy_bellman', 'no', 'wrong')
    jig_gate = g[(g.dataset == 'jigsaw') & (g.seed == 1601)]
    assert len(jig_gate) >= 1, 'missing Jigsaw gate rows'
    jig_promoted = bool(jig_gate.promote.any())
    guarded_status = 'promotes a challenger on the reported seed' if jig_promoted else 'remains the incumbent on the reported seed'
    zh_guarded = '已在報告機組上推出挑戰者' if jig_promoted else '在報告機組上仍維持原策略'
    text=f'''# Answer-aware routing optimization — v1.6 candidate

Experiment commit: `{v['tested_commit']}`. Hosted evidence: {v['run_url']}.
This is an algorithm extension on top of v1.5, not a new published-paper version or a fresh LLM benchmark.

## Algorithm changes

Preserve answer direction for normalized yes/no tasks instead of merging yes-majorities and no-majorities into one equality pattern. The richer joint model retains correlated errors rather than treating votes as independent.

A two-bin prompt-conditioned model fits representation and risk-table data separately, uses empirical-Bayes shrinkage, and falls back globally below 32 calibration cases. It is a small transparent estimator, not Jev or CERA.

Finite decisions compile into version-bound lookup tables. Loading validates the full state space, acquired-only terminal actions, query budget, costs and worker identities. JSON is data-only, not pickle. Check the trusted export hash before deployment: schema validation alone does not authenticate an artifact.

Tuning is separate from an independent promotion gate. A single tuning-selected challenger must pass a paired empirical-Bernstein upper bound before replacing the incumbent. Exact compilation of the incumbent requires no statistical improvement claim.

## Main results

Six existing binary datasets, 1,732 original test cases; c=0.01 per worker response, wrong loss=1, defer=0.25, at most three responses. Costs are assumed loss units, not dollars. Context-head CPU cost is not in J. Old test outcomes were previously inspected; this is exploratory repeated-benchmark optimization, not external confirmation. The legacy baseline is fitted on the original training split, with only its strength chosen on the tuning subset; it is not given gate data.

{table(rows,['method','objective','coverage','queries'])}

### Per-dataset loss

{table(tasks,['dataset','legacy_bellman','polarity_bellman','context_polarity'])}

Direction-aware states only apply to StrategyQA/Jigsaw. Other tasks preserve equality. Context uses part of training for representation and the remainder for conditional tables; global legacy/polarity tables use the full training budget. No candidate receives test gold or unacquired worker answers. GQA still lacks images.

### Minority-class diagnostic (post-primary)

{class_table}

For Jigsaw, yes is toxic. The average gain largely removes false positives among {nontoxic_n} non-toxic cases (legacy {legacy_nontoxic_wrong} wrong, polarity {new_nontoxic_wrong['polarity_bellman']} wrong, context+polarity {new_nontoxic_wrong['context_polarity']} wrong), while the new candidates correctly return a toxic answer in {new_toxic_correct['polarity_bellman']} and {new_toxic_correct['context_polarity']} of {toxic_n} positive cases respectively. This prevents an unqualified detection-quality claim. Guarded deployment {guarded_status}. A deployment valuing missed toxicity more strongly needs an explicit asymmetric loss and a separately validated policy. Changing scoring weights after seeing results is not presented as a primary experiment. This audit uses the same archive gold and changes no policies.

### Paired uncertainty

Differences are candidate minus legacy; negative is better. The 95% and conservative four-comparison 98.75% intervals condition on fitted policies. They do not remove benchmark reuse or incorporate retraining uncertainty.

{table(paired,list(paired.columns))}

### Selection sensitivity

{table(m,['seed','method','objective','coverage','queries'])}

Seeds change representation/calibration/tuning/gate assignments, not the original test set. They are not three independent external replications.

## Promotion is a separate question

New challengers promoted: **{int(g.promote.sum())}/{len(g)}** dataset/seed configurations. For D=J_new-J_old in [-M,M], M=1.03 and n independent gate groups, the one-candidate gate uses

`U = mean(D) + sqrt(2 sample_var(D) log(2/alpha)/n) + 7 (2M) log(2/alpha)/(3(n-1))`.

Promote only if U<0. Alpha=0.05 is per-dataset under iid groups and stationary deployment; it is conservative for small gates, not a simultaneous six-task guarantee, not robust to arbitrary drift, and cannot certify an adaptively overused benchmark. Rejecting retains the compiled incumbent. Do not treat exploratory test gains as automatic deployment authorization.

## Exact-behavior runtime improvement

{table(timing,['dataset','legacy','compiled','speedup'])}

Amortized nanoseconds per callback execution; interleaved order, same process, 30 rounds, 64 cases, warmup excluded. This is not LLM latency and excludes fitting/compilation. Both implementations return identical workers/queries/actions for all 52 complete equality/INVALID patterns across 18 fitted incumbents ({v['full_pattern_equivalence_cases']} checks), and on actual held-out cases. Rounded JSON risks may change last-decimal diagnostics, not precompiled actions. Context exports also undergo load/action equivalence tests.

## Verification and use

{v['tests']} tests, zero failures/skips; two complete processes; {v['ledger_rows']} per-case method/seed records and {v['exported_policies']} validated JSON artifacts per run. Repeated records are not new questions. Gates independently recompute cost, source-label errors, gate decisions, sample sets, split isolation, serialized hashes and repeated outputs. Only timing columns are excluded.

Use `results/optimizer/policies/<dataset>-1601-compiled_legacy.json` for exact-behavior acceleration or `results/optimizer/policies/<dataset>-1601-guarded_selected.json` for gated output. Other files are experimental. Ordered model identities and pool revision must match. ContextRouter adds prompt features; CompiledPolicy needs only acquired-response callbacks.

No proprietary Jev inference, new worker generation, CERA training, new datasets or human preferences. Existing releases remain unchanged.

## References

Maurer and Pontil, Empirical Bernstein Bounds and Sample Variance Penalization, 2009, arXiv:0907.3740. The gate applies existing statistics, not a new universal safety theorem. https://arxiv.org/abs/0907.3740

scikit-learn, Common pitfalls / Data leakage. https://scikit-learn.org/stable/common_pitfalls.html
'''
    (D/'report.en.md').write_text(text)
    zh=f'''# 答案方向感知路由優化：v1.6 候選

來源 `{v['tested_commit']}`；Actions：{v['run_url']}。
這是在 v1.5 上增加演算法與執行器，不是新的 LLM 推論或正式論文 release。

## 改了什麼

保留是／否、毒性／非毒性的答案方向；依題目特徵分區估計聯合風險，使用全域平滑與稀疏回退。決策編譯為模型版本綁定的查表格式。選參與獨立更新門檻分離，沒有直接把測試最漂亮的結果設成預設。

## 主要結果

相同六資料集與1,732題原測試，錯答成本1、轉交0.25、取得答案0.01，最多三次。成本不是美元，未計入context head計算。

{table(rows,['method','objective','coverage','queries'])}

各任務結果：

{table(tasks,['dataset','legacy_bellman','polarity_bellman','context_polarity'])}

方向資訊只用於StrategyQA／Jigsaw。其他任務仍看相等模式；GQA沒有圖片證據。平均改善不代表每個任務改善。這些是曾使用過的benchmark，不能稱作新的獨立外部驗證。

## 正類退步：不能只看平均改善

{class_table}

Jigsaw的yes代表毒性。新候選減少{nontoxic_n}個非毒性案例的誤判（舊策略{legacy_nontoxic_wrong}個誤判，新候選分別{new_nontoxic_wrong['polarity_bellman']}與{new_nontoxic_wrong['context_polarity']}個），但在{toxic_n}個毒性案例中分別僅正確回傳{new_toxic_correct['polarity_bellman']}與{new_toxic_correct['context_polarity']}個；不能稱作偵測能力全面提升。預設更新{zh_guarded}。重視漏判的部署須先指定不對稱損失，再驗證策略，不得看到測試後改指標冒充原實驗。這是同一歸檔真值的事後診斷，未改動主要策略。

## 配對區間

差值為新減舊，負值較佳。一般95%及四對比調整98.75%區間條件於已擬合策略，不能消除測試集重用限制。

{table(paired,list(paired.columns))}

## 未自動替換預設

獨立gate核准 **{int(g.promote.sum())}/{len(g)}** 組更新。它對成對損失差建立empirical-Bernstein上界，只在上界小於0時核准。小型gate很保守；這不是否認測試改善，而是部署證據門檻不同。未核准時guarded_selected維持原策略，不為讓新方法上線而放寬門檻。每任務條件式alpha=0.05不代表六任務同時95%保證，也不涵蓋分布漂移。

## 不改答案的速度改善

{table(timing,['dataset','legacy','compiled','speedup'])}

單位為每題攤提奈秒；同程序交替量測callback，不含模型推論與離線擬合。18個策略各核對52種完整模式，共{v['full_pattern_equivalence_cases']}個等價檢查；JSON回讀後亦確認測試動作不變。這不是LLM推論加速倍數。

## 驗證與使用

{v['tests']}項測試、零失敗零跳過；完整重跑兩次；每次{v['ledger_rows']}筆方法／seed紀錄與{v['exported_policies']}份JSON策略。不同seed仍用同一原測試集，不能重複算新樣本。CI重算來源損失、查詢成本、gate、切分與序列化，計時另報。

優先用compiled_legacy作原行為加速，guarded_selected為保守更新結果；polarity_bellman／context_polarity是實驗候選。模型身分、順序或版本變動會拒絕舊策略。JSON不執行pickle；仍須驗證可信來源雜湊，schema合法不等於檔案可信。

舊release不變。本輪未呼叫專有Jev、未生成新工作模型答案、未訓練CERA，也未做真人偏好實驗。完整三切分表、公式與引用見英文報告及協議。
'''
    (D/'report.zh-TW.md').write_text(zh)
    for name in ('macro.csv','summary.csv','comparisons.csv','runtime.csv','promotion_gates.csv','fit_resources.csv','class_outcomes.csv'):
        shutil.copy(R/name,D/name)
    shutil.copy('verification/optimizer.json',D/'verification.json')

if __name__=='__main__':main()

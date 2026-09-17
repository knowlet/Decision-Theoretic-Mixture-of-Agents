"""Build evidence only after two complete replay/audit executions agree.

This is a computational reproducibility gate, not a test of scientific superiority.
No results are invented and no real-model endpoint is called by this program.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

REQUIRED = frozenset('derived_traces.npz common_error_diagnostics.csv cost_deferral_sweep.csv development_selection.csv five_shot_comparisons.csv five_shot_provenance.json five_shot_per_case.csv five_shot_summary.csv mmlu_gold_audit.json mmlu_gold_regraded_per_case.csv mmlu_gold_pairs.csv mmlu_gold_regraded_summary.csv ood_per_case.csv ood_summary.csv primary_by_family.csv primary_comparisons.csv primary_per_case.csv primary_summary.csv provenance.json test_error_correlations.csv split_manifest.csv'.split())
POLICIES = frozenset('bellman defer_all disagreement_gate fixed_three majority_three myopic single_best static_all_sources'.split())
VOLATILE = frozenset(['elapsed_seconds', 'python', 'platform'])

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare_runs(first: Path, second: Path) -> list[dict]:
    a = {p.name for p in first.iterdir() if p.is_file()}
    b = {p.name for p in second.iterdir() if p.is_file()}
    require(REQUIRED <= a, f'Missing outputs: {sorted(REQUIRED-a)}')
    require(a == b, f'Output sets differ: {sorted(a ^ b)}')
    records = []
    for name in sorted(a):
        pa, pb = first/name, second/name
        require(not pa.is_symlink() and not pb.is_symlink(), 'Symlink output is not allowed')
        ha, hb = sha256(pa), sha256(pb)
        if name == 'provenance.json':
            ja, jb = json.loads(pa.read_text()), json.loads(pb.read_text())
            ja = {k: v for k, v in ja.items() if k not in VOLATILE}
            jb = {k: v for k, v in jb.items() if k not in VOLATILE}
            require(ja == jb, f'Nonvolatile provenance differs: {name}')
            mode = 'JSON excluding elapsed_seconds, python, platform'
        elif name.endswith('.npz'):
            with np.load(pa, allow_pickle=False) as x, np.load(pb, allow_pickle=False) as y:
                require(set(x.files) == set(y.files), 'NPZ keys differ')
                for key in x.files:
                    require(x[key].dtype == y[key].dtype, 'NPZ dtype differs')
                    np.testing.assert_array_equal(x[key], y[key])
            mode = 'Exact array equality; pickle disabled'
        else:
            require(ha == hb, f'Output differs: {name}')
            mode = 'SHA-256 byte identity'
        records.append(dict(file=name, mode=mode, sha256_first=ha, sha256_second=hb, byte_identical=ha == hb))
    return records

def validate_primary(directory: Path) -> dict:
    frame = pd.read_csv(directory/'primary_per_case.csv')
    require(len(frame) > 0, 'Empty primary evaluation')
    require(set(frame.policy) == POLICIES, 'Missing or unexpected policies')
    require(not frame.duplicated(['sample_id', 'policy']).any(), 'Duplicate primary case/policy')
    sets = [set(g.sample_id) for _, g in frame.groupby('policy')]
    require(all(x == sets[0] for x in sets), 'Policies evaluated different cases')
    require(np.isfinite(frame[['terminal_loss','historical_cost_usd','objective','acquisitions']].to_numpy()).all(), 'Nonfinite objective/resource')
    require((frame.historical_cost_usd >= 0).all(), 'Negative cost')
    require(frame.acquisitions.between(0, 3).all(), 'Invalid query budget')
    require((frame.acquisitions == np.floor(frame.acquisitions)).all(), 'Fractional acquisition')
    require(frame.answered.isin([0, 1]).all(), 'Invalid coverage indicator')
    require((frame.answered == (frame.selected_model_index >= 0).astype(int)).all(), 'Coverage/index mismatch')
    require(frame.loc[frame.answered == 1, 'selected_score'].isin([0, 1]).all(), 'Invalid selected score')
    require(frame.loc[frame.answered == 0, 'selected_score'].isna().all(), 'Deferral counted as correct')
    expected = np.where(frame.answered == 1, 1-frame.selected_score, frame.defer_loss)
    np.testing.assert_allclose(frame.terminal_loss, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame.objective, expected + frame.lambda_usd*frame.historical_cost_usd, rtol=0, atol=1e-12)
    manifest = pd.read_csv(directory/'split_manifest.csv')
    require(not manifest.ids.duplicated().any(), 'Duplicate manifest IDs')
    require(manifest.groupby('groups').splits.nunique().max() == 1, 'Prompt group split leakage')
    require(sets[0] == set(manifest.loc[manifest.splits == 'test', 'ids']), 'Primary is not exactly the held-out split')
    p = json.loads((directory/'provenance.json').read_text())
    require(p['new_llm_api_calls'] == 0, 'Unexpected inference claim')
    require(p['audit']['cross_split_group_overlap'] == 0, 'Overlap recorded in provenance')
    require(p['audit']['retained_rows'] == len(manifest), 'Manifest count differs')
    require(p['audit']['split_counts']['test'] == len(sets[0]), 'Test count differs')
    # Independently recompute main reported summaries from the per-case ledger.
    summary = pd.read_csv(directory/'primary_summary.csv').set_index('policy')
    for name, g in frame.groupby('policy'):
        row = summary.loc[name]
        require(row['n'] == len(g), 'Reported n differs')
        for field in ['objective','terminal_loss','historical_cost_usd','acquisitions']:
            np.testing.assert_allclose(row[field], g[field].mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(row.coverage, g.answered.mean(), rtol=0, atol=1e-12)
    gold = json.loads((directory/'mmlu_gold_audit.json').read_text())
    require(gold['matched_rows'] >= .95*gold['original_mmlu_rows'], 'Insufficient gold matching')
    pairs = pd.read_csv(directory/'mmlu_gold_pairs.csv')
    require(len(pairs) == gold['evaluated_model_response_pairs'], 'Gold pair count differs')
    require(pairs.sample_id.nunique() == gold['matched_rows'], 'Gold item count differs')
    require(int(pairs.score_difference.sum()) == gold['all_strict_score_disagreements'], 'Gold disagreement count differs')
    return dict(test_rows=len(sets[0]), retained_rows=len(manifest), policies=len(POLICIES),
                primary_model_responses=len(sets[0])*4, archived_model_responses=len(manifest)*4)

def test_counts(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    require(len(cases) > 0 and failed == 0 and skipped == 0, 'Tests empty, failing or skipped')
    return dict(tests=len(cases), failures=failed, skipped=skipped)

def table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append('NA' if pd.isna(value) else (f'{value:.6f}' if isinstance(value, (float, np.floating)) else str(value)))
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)

def reports(directory: Path, evidence: dict, out: Path) -> None:
    primary = pd.read_csv(directory/'primary_summary.csv')
    pairs = pd.read_csv(directory/'primary_comparisons.csv')
    shift = pd.read_csv(directory/'five_shot_summary.csv')
    p = json.loads((directory/'provenance.json').read_text())
    gold = json.loads((directory/'mmlu_gold_audit.json').read_text())
    five = json.loads((directory/'five_shot_provenance.json').read_text())
    columns = ['policy','objective','historical_cost_usd','coverage','selective_accuracy','acquisitions']
    primary_table = table(primary, columns)
    pair_table = table(pairs, ['a','b','difference','bootstrap_lo','bootstrap_hi','holm_p'])
    shift_table = table(shift, columns)
    counts = p['audit']['split_counts']
    common = f'''\n## Provenance\n\nTested commit: `{evidence['tested_commit']}`. [Hosted execution]({evidence['run_url']}).\n\nRouterBench source revision: `{p['source_revision']}`.\nZero-shot SHA-256: `{p['source_sha256']}`.\nFive-shot SHA-256: `{five['source_sha256']}`.\n\nModels: {', '.join('`'+m+'`' for m in p['models'])}.\n\n## Primary results\n\n{primary_table}\n\n## Paired prompt-group uncertainty\n\n{pair_table}\n\n## Frozen zero-to-five-shot stress test\n\n{shift_table}\n'''
    english = f'''# Historical LLM Replay: Hosted Execution and Validity Audit\n\nTechnical-report addendum, 2026-09-17. Not peer reviewed.\n\n## Findings\n\nReal historical LLM outputs replace synthetic correctness channels in this evidence layer. This is offline selection among archived responses, not fresh inference, inter-agent debate, or free-text synthesis. The primary Bellman-minus-myopic and Bellman-minus-static intervals include zero: the replay does not establish a reliable advantage over those strong controls. It is not an equivalence or noninferiority test.\n\nThe existing primary analysis was observed before this execution-hardening change. No hypotheses, model pool, losses, parser, split, calibration, or primary scores were altered by the CI/report revision. The second execution reruns training, development selection, all cost/deferral settings, held-out-family analyses, independent-reference regrading and prompt-shift replay; it is not merely reevaluating one cached policy.\n\n## Methods and evidence scope\n\nThe frozen source contains {p['audit']['source_rows']:,} rows; the four retained benchmark families contribute {p['audit']['retained_rows']:,} rows ({evidence['archived_model_responses']:,} model-response records). Train/dev/test counts are {counts['train']:,}/{counts['dev']:,}/{counts['test']:,}. Exact normalized prompts are grouped before deterministic splitting. There are {p['audit']['duplicate_prompt_rows']} duplicate-prompt rows; no exact group crosses a split. Near-duplicate or semantic leakage is not excluded.\n\nFour models are available to every policy, with a three-acquisition ceiling. Primary loss is wrong-answer loss (1), deferral loss (0.25), plus 10 times historical estimated USD. Deferral cost and the USD multiplier are study choices, not measured human utility. Runtime controller selection is restricted to paid-for answer categories. Costs, labels and correctness are not policy observations before acquisition; full archived traces remain in the evaluator. Hyperparameter selection uses development labels, not test labels. Other than the diagnostic non-abstaining majority control, methods share the terminal selection rule.\n\nThe paired 95% percentile bootstrap resamples normalized-prompt groups with 2,000 replicates, conditional on fitted policies. Holm p-values use a cluster-t approximation, not the percentile-bootstrap distribution. Small negative point estimates are not a superiority claim when intervals include zero. These four task families and four historical models are not a representative sample of all MoA applications.\n\n## Independent-reference audit and transport limits\n\nMMLU gold matches {gold['matched_rows']:,}/{gold['original_mmlu_rows']:,} archived questions. It independently checks {gold['evaluated_model_response_pairs']:,} response/label pairs; strict-score disagreements: {gold['all_strict_score_disagreements']}. Correct option TEXT is remapped when choices are reordered ({gold['reordered_choice_matches']:,} questions), rather than reusing the reference letter. This does not certify gold correctness, noncontamination or the other benchmark families.\n\nThe frozen five-shot test matches {five['matched_test_rows']:,} original test questions. It does not retrain policies, probabilities, parser or expected prices. Accuracy, formatting and prompt-length costs all change, so this is a compound prompt-format/cost stress test, not isolated distribution shift. Demonstration overlap was not audited. Notably, Claude-v2 has {five['five_shot_loading_audit']['invalid_answer_count_by_model']['claude-v2']:,} strict-parser INVALID outputs among {five['five_shot_loading_audit']['retained_rows']:,} eligible five-shot rows; INVALID is retained, not silently dropped.\n\n## Reproducibility\n\n{evidence['tests']} tests passed without failures or skips. Two complete runs agree for {len(evidence['outputs'])} required outputs under the recorded comparison rules. Data CSV/JSON files must be byte-identical; NPZ arrays must be exactly equal. Only execution time and runtime metadata in provenance may differ. Read `verification.json` for per-file hashes. This is same-code computational reproducibility on a GitHub-hosted runner, not external scientific replication.\n\n## What remains unvalidated\n\nThere are zero new LLM API calls; historical estimated USD is not a present-day invoice. Service latency, tool behavior, action-dependent prompting, natural-language synthesis, human preferences, subjective tradeoffs and current frontier-model performance remain unvalidated. The synthetic Bellman theorem does not prove superiority of this fitted archive-based controller. Full-table state estimation and its errors may favor simpler policies.\n''' + common + '''\n## Sources\n\n- Hu et al. (2024), ROUTERBENCH: A Benchmark for Multi-LLM Routing System, arXiv:2403.12031. https://arxiv.org/abs/2403.12031\n- RouterBench official code: https://github.com/withmartian/routerbench\n- Frozen data: https://huggingface.co/datasets/withmartian/routerbench\n- MMLU reference: https://huggingface.co/datasets/cais/mmlu\n\nUpstream terms apply to upstream data. Distributed derived traces omit raw prompt/response text. Existing project licensing is unchanged. AI assisted implementation and reporting; no invented reviewers or institutional claims.\n'''
    chinese = f'''# 真實 LLM 歷史重播：GitHub Actions 執行與效度稽核\n\n2026-09-17 技術報告附錄；未經同儕審查。\n\n## 核心結論\n\n這次使用 RouterBench 實際保存的 LLM 回答，而不是 Bernoulli 錯誤通道或傳統分類器替身。但它仍是離線選擇既有答案，不是新呼叫模型、多代理互相辯論或自然語言綜合。Bellman 相對貪婪級聯及同權限靜態批次的配對信賴區間均跨零，不能宣稱可靠勝出；也不能據此宣稱兩者等價。\n\n遠端先前已有主分析結果。本次 CI 強化沒有改動假說、模型池、損失、解析器、切分或主要分數；完整重跑兩次訓練、開發集選參、成本／保留判斷掃描、留出任務族、獨立參考答案稽核及五示例壓力測試。\n\n## 實驗規模與公平性\n\n原資料 {p['audit']['source_rows']:,} 筆；四個任務族保留 {p['audit']['retained_rows']:,} 題、共 {evidence['archived_model_responses']:,} 筆模型回答紀錄。訓練／開發／測試為 {counts['train']:,}／{counts['dev']:,}／{counts['test']:,}。正規化後完全相同的題目不跨切分；沒有排除語意相近、近重複或預訓練污染。\n\n所有策略可存取同一組四模型，最多取得三次答案。主要目標為錯答損失 1、保留判斷損失 0.25，再加上歷史估算美元乘以 10。這些效用權重是研究設定，不是實測使用者偏好。除診斷用、不保留判斷的多數決外，各策略共用終端選擇規則。測試標籤不參與策略擬合或選參。\n\n主比較以題目群組做 2,000 次配對 bootstrap，95% 區間條件於已擬合策略，不包含獨立重訓的不確定性；Holm 校正的 p 值來自群組 t 近似。\n\n## 答案稽核與限制\n\nMMLU 獨立參考答案對上 {gold['matched_rows']:,}／{gold['original_mmlu_rows']:,} 題，核對 {gold['evaluated_model_response_pairs']:,} 組回答與標籤，嚴格評分差異 {gold['all_strict_score_disagreements']}。其中 {gold['reordered_choice_matches']:,} 題需依正確選項文字重新映射順序；不能沿用原字母。這不代表 gold 本身一定正確，也沒有排除訓練污染。\n\n五示例測試保留 {five['matched_test_rows']:,} 題，策略不重訓；答案格式、品質及歷史成本同時改變，應稱複合壓力測試。Claude-v2 在全部合格五示例資料有 {five['five_shot_loading_audit']['invalid_answer_count_by_model']['claude-v2']:,} 筆無法被嚴格前綴解析器識別，仍以 INVALID 保留，沒有排除。示例與校準題的重疊尚未稽核。\n\n## 執行驗證\n\n{evidence['tests']} 項測試通過，零失敗、零跳過；兩次完整執行的 {len(evidence['outputs'])} 個必要結果檔通過規定的核對。CSV 等資料要求逐位元相同；NPZ 要求每個陣列精確一致；僅 provenance 的耗時及執行環境欄位允許不同。這是同程式重現，不是外部研究者的科學複現。\n\n仍未驗證自由文字綜合、使用者價值分歧、工具呼叫、即時推論、服務延遲及目前模型能力。新 LLM API 呼叫為零；原先合成問題的 Bellman 最優性不會自動轉成真實控制器的普遍優勢。\n''' + common + '\n完整方法、來源與限制另見英文附錄。README 的原 v1.1 論文仍保留，不把尚未做過的實驗回填成既有證據。\n'
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.en.md').write_text(english)
    (out/'report.zh-TW.md').write_text(chinese)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=Path, default=Path('results/external'))
    parser.add_argument('--second', type=Path, default=Path('results/external-repeat'))
    parser.add_argument('--junit', type=Path, default=Path('verification/all-tests.xml'))
    parser.add_argument('--output', type=Path, default=Path('dist-external'))
    args = parser.parse_args()
    outputs = compare_runs(args.first, args.second)
    counts = validate_primary(args.first)
    validate_primary(args.second)
    tests = test_counts(args.junit)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    require(not os.getenv('GITHUB_SHA') or os.environ['GITHUB_SHA'] == commit, 'Checkout does not match triggering commit')
    evidence = dict(schema_version=1, tested_commit=commit,
        run_url=os.getenv('RESEARCH_RUN_URL','local execution'),
        runner_os=platform.platform(), python=sys.version, **tests, **counts,
        complete_executions=2, all_checks_passed=True, new_llm_api_calls=0, outputs=outputs,
        scope='Historical LLM response replay; not fresh inference, product comparison, or external scientific replication.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'verification.json').write_text(json.dumps(evidence, indent=2)+'\n')
    reports(args.first, evidence, args.output)
    print(json.dumps({k:v for k,v in evidence.items() if k != 'outputs'}, indent=2))

if __name__ == '__main__':
    main()

"""Result-derived full bilingual paper; reporting and inference commits are distinct."""
import json,os,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import build_transfer_paper
R=Path('results/openjev');OUT=Path('dist-paper')
NAMES={'empirical_single':'Single','empirical_fixed3':'Fixed-3','empirical_static':'Static','empirical_myopic':'Myopic','empirical_bellman':'Bellman','empirical_disagreement':'Disagreement','empirical_majority3':'Top-3 vote','empirical_prompt_top1':'Prompt top-1','empirical_cumulative_score':'Cumulative','matched_majority3':'Panel vote','openjev_raw_fixed3':'OpenJev raw','openjev_calibrated_fixed3':'OpenJev calibrated'}

def tab(df,cols,labels=None):
    lines=['| '+' | '.join(labels or cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
    for row in df.to_dict('records'):
        values=[]
        for c in cols:
            v=row[c]
            if pd.isna(v):s='—'
            elif isinstance(v,float):s=f'{v:.5f}'
            else:s=str(v)
            values.append(s.replace('|','/'))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)

def build():
    OUT.mkdir(exist_ok=True)
    verify=json.loads((R/'verification.json').read_text())
    if not verify['all_gates_passed']:raise RuntimeError('No verified outcomes')
    summary=pd.read_csv(R/'summary.csv');comp=pd.read_csv(R/'comparisons.csv')
    calibration=pd.read_csv(R/'calibration.csv');robust=pd.read_csv(R/'robustness.csv')
    chosen=summary[(summary.model=='qwen35-4b')|summary.algorithm.str.startswith('openjev_')].copy()
    chosen['algorithm']=chosen.algorithm.map(NAMES);chosen['model']=chosen.model.str.replace('qwen35-','',regex=False)
    paired=comp[comp.dataset=='macro_binary'].copy()
    paired['contrast']=['raw vs fixed-3','calibrated vs fixed-3','calibrated vs Bellman','0.8B raw vs 4B raw']
    confidence=calibration.copy();confidence['score']=confidence.score.replace({'raw_option_score':'Raw proxy','calibrated_success':'Dev-calibrated'});confidence['model']=confidence.model.str.replace('qwen35-','',regex=False)
    robustness=robust.copy();robustness['model']=robustness.model.str.replace('qwen35-','',regex=False)
    timed=chosen[(chosen.model=='4b')&chosen.algorithm.isin(['Fixed-3','Bellman','OpenJev raw','OpenJev calibrated'])]
    values={'SCORE_TABLE':tab(chosen,['model','algorithm','objective','coverage','accuracy_on_answered','queries'],['Model','Algorithm','J','Coverage','Accuracy*','Queries']),
       'PAIRS':tab(paired,['contrast','difference','lo','hi'],['Contrast','Difference','95% low','95% high']),
       'FAMILY':tab(paired,['contrast','family_lo','family_hi'],['Contrast','98.75% low','98.75% high']),
       'CALIBRATION':tab(confidence,['model','score','brier','ece'],['Model','Score','Brier','ECE']),
       'ROBUSTNESS':tab(robustness,['model','n','flips','mean_tv'],['Model','n','Order flips','Mean TV']),
       'TIMING':tab(timed,['algorithm','controller_seconds_median','controller_seconds_p95','input_tokens_mean'],['Algorithm','Median seconds','p95 seconds','Mean tokens']),
       'EXPERIMENT_COMMIT':verify['tested_commit'],'EXPERIMENT_RUN':verify.get('run_url') or 'unavailable','TOTAL_FORWARDS':str(verify['total_local_selector_forwards'])}
    tests=[]
    for path in sorted(Path('incoming').rglob('*tests.xml')):
        cases=list(ET.parse(path).getroot().iter('testcase'))
        if not cases or any(c.find(k) is not None for c in cases for k in ('failure','error','skipped')):raise RuntimeError('Unverified upstream/adapter tests')
        tests.append({'suite':str(path),'tests':len(cases),'failures':0,'skipped':0})
    root_tests=list(ET.parse('verification/transfer-tests.xml').getroot().iter('testcase'))
    if not root_tests or any(c.find(k) is not None for c in root_tests for k in ('failure','error','skipped')):raise RuntimeError('Unverified regression suite')
    tests.append({'suite':'combined regression','tests':len(root_tests),'failures':0,'skipped':0})
    build_transfer_paper.build()
    for lang in ('en','zh-TW'):
        text=Path(f'dist-transfer/paper.{lang}.md').read_text().replace('Version 1.3.0','Version 1.4.0').replace('v1.3.0 ·','v1.4.0 ·')
        if lang=='en':
            text=text.replace("title: 'When to Update the Router'","title: 'When to Update the Router—and When to Use a Learned Selector'")
            text=text.replace('No Jev inference, CERA-MoA agent training, live LLM generation, or human preference experiment was performed.',
                'The population-transfer layer performs no proprietary Jev inference, CERA-MoA training, new worker generation, or human preference experiment. A separately specified head-to-head extension below executes local OpenJev selector inference on a 128-question pilot with 64 development cases. Matched worker-panel and policy-level comparisons are reported separately, including actual selector CPU time and option-order sensitivity.')
            heading='# References and attribution'
        else:
            text=text.replace("title: '何時該更新路由器？'","title: '何時更新路由器，何時使用學習式選擇器？'")
            text=text.replace('本研究沒有執行 Jev 推論、CERA-MoA 代理訓練、新 LLM 生成或人類偏好實驗。',
                '模型池替換部分沒有執行專有 Jev 推論、CERA-MoA 訓練、新工作模型生成或人類偏好實驗。本版另加入實際 OpenJev 選擇器推論：以 128 題留出 pilot 與 64 題開發資料，比較相同候選資訊下的答案選擇，並分開完整策略、選擇器 CPU 耗時與選項順序敏感度。')
            heading='# 參考文獻、揭露與資料歸屬'
        chapter=Path(f'openjev_chapter.{lang}.md').read_text()
        for k,v in values.items():chapter=chapter.replace('@@'+k+'@@',v)
        if '@@' in chapter:raise RuntimeError('Unfilled result placeholder')
        if heading not in text:raise RuntimeError('Manuscript join point changed')
        text=text.replace(heading,chapter+'\n\n'+heading)
        text=re.sub(r'(?<!<)https?://[^\s<>。]+',lambda m:'<'+m.group().rstrip('.,')+'>'+m.group()[len(m.group().rstrip('.,')):],text)
        (OUT/f'paper.{lang}.md').write_text(text)
    provenance={'version':'1.4.0','experiment_commit':verify['tested_commit'],'experiment_run':verify['run_url'],
        'reporting_commit':os.getenv('REPORTING_COMMIT','local-render'),'reporting_run':os.getenv('REPORTING_RUN_URL','local-render'),
        'experiment_evidence':verify,'test_suites':tests,'scope':'Reporting reuses verified inference outputs. Its reporting commit is not relabeled as the inference commit.'}
    (OUT/'verification.json').write_text(json.dumps(provenance,indent=2)+'\n')
    for name in ('summary.csv','comparisons.csv','calibration.csv','robustness.csv','selector_cost_sensitivity.csv','by_dataset.csv'):shutil.copy(R/name,OUT/name)
    print('Generated manuscripts from experiment',verify['tested_commit'])

if __name__=='__main__':build()

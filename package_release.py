"""Package only audited study artifacts; never bundle installed fonts or credentials."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,zipfile
ROOT=Path(__file__).resolve().parent
att=json.loads((ROOT/'verification/attestation.json').read_text())
assert att['tests']==86 and att['revision_complete'] and att['all_numerical_comparisons_passed']
if os.environ.get('RESEARCH_RUN_URL'):att['github_actions_run_url']=os.environ['RESEARCH_RUN_URL']
(ROOT/'verification/attestation.json').write_text(json.dumps(att,indent=2)+'\n')
dist=ROOT/'dist';dist.mkdir(exist_ok=True)
for src,name in [(ROOT/'paper.en.pdf','paper.en.pdf'),(ROOT/'paper.zh-TW.pdf','paper.zh-TW.pdf'),
                 (ROOT/'results/summary.csv','primary-results.csv'),(ROOT/'results/revision/matched_summary.csv','matched-results.csv'),
                 (ROOT/'verification/attestation.json','verification.json')]:shutil.copy2(src,dist/name)
files=[]
for f in sorted(ROOT.rglob('*')):
    if not f.is_file():continue
    rel=f.relative_to(ROOT)
    if any(part.startswith('.') or part=='__pycache__' for part in rel.parts):continue
    if rel.parts[0] in ('dist',):continue
    if f.suffix in ('.aux','.log','.toc','.out','.pyc','.synctex','.gz'):continue
    if f.suffix not in ('.py','.md','.tex','.pdf','.json','.csv','.npz','.png','.sh','.txt','.cff','.xml'):continue
    files.append(f)
with zipfile.ZipFile(dist/'reproducibility-package.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in files:z.write(f,str(f.relative_to(ROOT)))
lines=[]
for f in sorted(dist.iterdir()):
    if f.is_file() and f.name!='SHA256SUMS':lines.append(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name)
(dist/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
print('Packaged',len(files),'research files; no model endpoints or font files included.')

"""Package verified derived artifacts, excluding upstream text and font files."""
import hashlib,json,subprocess,zipfile
from pathlib import Path
root=Path('.');dist=Path('dist-transfer')
verification=json.loads((dist/'verification.json').read_text())
if not verification['all_gates_passed'] or verification['failures'] or verification['skipped']:raise RuntimeError('Unverified release')
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if verification['tested_commit']!=commit:raise RuntimeError('Commit differs from executed source')
source=dist/'source.tar.gz'
subprocess.run(['git','archive','--format=tar.gz',f'--output={source}','HEAD'],check=True)
with zipfile.ZipFile(dist/'reproducibility.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.write(source,'source.tar.gz')
    for folder in (Path('results/transfer'),Path('results/transfer-repeat'),Path('verification')):
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,str(p))
    for lang in ('en','zh-TW'):
        for suffix in ('md','pdf','tex'):
            p=dist/f'paper.{lang}.{suffix}';z.write(p,p.name)
source.unlink()
# Keep distribution lean; TeX build intermediates stay inside source/log records.
for pattern in ('*.aux','*.log','*.out'):
    for p in dist.glob(pattern):p.unlink()
files=[p for p in sorted(dist.iterdir()) if p.is_file() and p.name!='SHA256SUMS']
(dist/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
print('Packaged',len(files),'artifacts from',commit)

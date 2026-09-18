#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python report_openjev.py
python - <<'PY'
from pathlib import Path
s=Path('header.tex').read_text().replace('Synthetic study · 2026-09-17','Routing and selection · 2026-09-18')
Path('report-header.tex').write_text(s)
PY
COMMON=(--standalone -V documentclass=article -V papersize=a4 -V fontsize=11pt
 -V 'geometry:margin=22mm' -V linestretch=1.08
 -V 'mainfont=Liberation Serif' -V 'sansfont=Liberation Sans'
 -V 'monofont=DejaVu Sans Mono' -V 'CJKmainfont=Noto Serif CJK TC'
 -V 'CJKsansfont=Noto Sans CJK TC' -H report-header.tex)
for language in en zh-TW; do
  pandoc "dist-paper/paper.$language.md" "${COMMON[@]}" -o "dist-paper/paper.$language.tex"
  python - "$language" <<'PY'
import re,sys
from pathlib import Path
p=Path('dist-paper')/f'paper.{sys.argv[1]}.tex';s=p.read_text()
s=re.sub(r'\\texttt\{([0-9a-f]{40,64})\}',lambda m:r'\texttt{'+r'\allowbreak{}'.join(m.group(1)[i:i+8] for i in range(0,len(m.group(1)),8))+'}',s)
p.write_text(s)
PY
  for pass in 1 2; do
    if ! xelatex -interaction=nonstopmode -halt-on-error -output-directory=dist-paper "dist-paper/paper.$language.tex" > "verification/openjev-latex-$language-$pass.log" 2>&1; then
      tail -90 "verification/openjev-latex-$language-$pass.log"; exit 1
    fi
  done
  if grep -q 'Missing character:' "verification/openjev-latex-$language-2.log"; then echo "Missing glyph $language"; exit 1; fi
  test -s "dist-paper/paper.$language.pdf"
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python build_transfer_paper.py
COMMON=(--standalone -V documentclass=article -V papersize=a4 -V fontsize=11pt
 -V 'geometry:margin=22mm' -V linestretch=1.08
 -V 'mainfont=Liberation Serif' -V 'sansfont=Liberation Sans'
 -V 'monofont=DejaVu Sans Mono' -V 'CJKmainfont=Noto Serif CJK TC'
 -V 'CJKsansfont=Noto Sans CJK TC' -H header.tex)
for language in en zh-TW; do
  pandoc "dist-transfer/paper.$language.md" "${COMMON[@]}" -o "dist-transfer/paper.$language.tex"
  python - "$language" <<'PYCODE'
import re,sys
from pathlib import Path
p=Path('dist-transfer')/f'paper.{sys.argv[1]}.tex'
s=p.read_text()
s=re.sub(r'\\texttt\{([0-9a-f]{40,64})\}',lambda m:r'\texttt{'+r'\allowbreak{}'.join(m.group(1)[i:i+8] for i in range(0,len(m.group(1)),8))+'}',s)
p.write_text(s)
PYCODE
  for pass in 1 2; do
    if ! xelatex -interaction=nonstopmode -halt-on-error -output-directory=dist-transfer "dist-transfer/paper.$language.tex" > "verification/transfer-latex-$language-$pass.log" 2>&1; then
      tail -90 "verification/transfer-latex-$language-$pass.log"; exit 1
    fi
  done
  if grep -q 'Missing character:' "verification/transfer-latex-$language-2.log"; then echo "Missing glyph $language"; exit 1; fi
  test -s "dist-transfer/paper.$language.pdf"
done

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
python make_figures.py
python make_revision_figures.py
python build_paper.py
python build_revision_tables.py
COMMON=(--standalone --number-sections
  -V documentclass=article -V papersize=a4 -V fontsize=11pt
  -V 'geometry:margin=22mm' -V linestretch=1.1
  -V 'mainfont=Liberation Serif' -V 'sansfont=Liberation Sans'
  -V 'monofont=DejaVu Sans Mono'
  -V 'CJKmainfont=Noto Serif CJK TC' -V 'CJKsansfont=Noto Sans CJK TC'
  -H header.tex)
for language in zh-TW en; do
  pandoc "paper.$language.md" "${COMMON[@]}" --pdf-engine=xelatex -o "paper.$language.tex"
  for pass in 1 2 3; do
    xelatex -interaction=nonstopmode -halt-on-error "paper.$language.tex" >> "latex_build.$language.log" 2>&1
  done
  test -s "paper.$language.pdf"
done
python - <<'PY'
from pathlib import Path
for lang in ('en','zh-TW'):
    log=Path(f'latex_build.{lang}.log').read_text(errors='replace')
    assert 'Missing character:' not in log, f'Missing glyphs: {lang}'
    assert '@@' not in Path(f'paper.{lang}.md').read_text()
    print(lang, 'PDF built:', Path(f'paper.{lang}.pdf').stat().st_size, 'bytes')
PY

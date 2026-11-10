#!/bin/sh
# Typeset THEORY.md as THEORY.pdf.  Requires pandoc and XeLaTeX.
#   sh build_pdf.sh                (run from the repository root)
set -e
cd "$(dirname "$0")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/header.tex" <<'EOF'
\usepackage{amsmath,amssymb}
\usepackage[margin=1.1in]{geometry}
\usepackage{microtype}
\setlength{\parskip}{0.5em}
\setlength{\emergencystretch}{3em}
\usepackage{newunicodechar}
\newunicodechar{∎}{\ensuremath{\blacksquare}}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Rational reconstruction from decimal expansions}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.2pt}
EOF
pandoc THEORY.md -o THEORY.pdf \
  --pdf-engine=xelatex --toc --toc-depth=2 \
  --shift-heading-level-by=-1 \
  --include-in-header="$TMP/header.tex" \
  --metadata subtitle="Notes accompanying the ratrecon library" \
  --metadata author="Raphael Neville" \
  -V documentclass=article -V fontsize=11pt \
  -V linkcolor=blue -V urlcolor=blue -V colorlinks=true
echo "wrote THEORY.pdf"

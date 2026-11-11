#!/bin/bash

# Ensure output directory exists
mkdir -p build

echo "Compiling Markdown sections into PDF..."

pandoc metadata.yaml \
    sections/*.md \
    --template=template.tex \
    --pdf-engine=xelatex \
    --top-level-division=chapter \
    --toc \
    --number-sections \
    --no-highlight \
    --citeproc \
    -o build/final_technical_report.pdf

echo "Build complete. Check build/final_technical_report.pdf"
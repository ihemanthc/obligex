#!/usr/bin/env python3
"""Convert a `# %%` percent-format Python file into a .ipynb notebook.

Cell markers:
    # %% [markdown]   -> markdown cell (leading '# ' stripped from each line)
    # %%              -> code cell
    # %% [code]       -> code cell

Also syntax-checks every code cell (ignoring IPython !shell and %magic lines)
so a broken notebook can never be shipped.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

MARKER = re.compile(r"^#\s*%%\s*(\[(?P<kind>[a-zA-Z]+)\])?\s*(?P<title>.*)$")


def parse(src: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    kind = "code"
    buf: list[str] = []
    for line in src.splitlines():
        m = MARKER.match(line)
        if m:
            if buf:
                cells.append((kind, "\n".join(buf).strip("\n")))
            buf = []
            kind = (m.group("kind") or "code").lower()
            continue
        buf.append(line)
    if buf:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, s) for k, s in cells if s.strip()]


def strip_md(text: str) -> str:
    out = []
    for line in text.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def checkable(code: str) -> str:
    """Blank out IPython-only lines so ast.parse can validate the rest."""
    out = []
    for line in code.splitlines():
        s = line.lstrip()
        if s.startswith("!") or s.startswith("%"):
            out.append("pass" + " " * 0)
        else:
            out.append(line)
    return "\n".join(out)


def build(path: Path) -> dict:
    cells_src = parse(path.read_text(encoding="utf-8"))
    nb_cells = []
    errors = []
    n_code = 0
    for idx, (kind, body) in enumerate(cells_src):
        if kind == "markdown":
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [l + "\n" for l in strip_md(body).splitlines()],
                }
            )
        else:
            n_code += 1
            try:
                ast.parse(checkable(body))
            except SyntaxError as e:
                errors.append(f"  cell {idx} (code #{n_code}) line {e.lineno}: {e.msg}")
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [l + "\n" for l in body.splitlines()],
                }
            )
    if errors:
        print(f"SYNTAX ERRORS in {path.name}:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)

    return {
        "cells": nb_cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    total_code = 0
    for arg in sys.argv[1:]:
        src = Path(arg)
        nb = build(src)
        out = src.with_suffix(".ipynb")
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        total_code += n_code
        print(f"{out.name}: {n_code} code + {n_md} markdown cells, syntax OK")
    print(f"total code cells validated: {total_code}")

"""Render the mccd-vs-psfex fiber report: template.html with {{img:NAME}} and
{{key}} placeholders, images base64-inlined, numbers from summary.json.

Usage: python build_report.py template.html comparison_dir out.html
"""

import base64
import json
import re
import sys
from pathlib import Path

tpl, comp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
html = tpl.read_text()


def img(m):
    data = base64.b64encode((comp / m.group(1)).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


html = re.sub(r"\{\{img:([\w.\-]+)\}\}", img, html)

summary_path = comp / "summary.json"
if summary_path.exists():
    s = json.loads(summary_path.read_text())

    def num(m):
        cur = s
        for k in m.group(1).split("|"):
            cur = cur[k]
        return f"{cur:{m.group(2) or '.3f'}}"

    html = re.sub(r"\{\{num:([^}:]+):?([^}]*)\}\}", num, html)
out.write_text(html)
print(out)

"""Validate local links, source-card paths and syntax. No network claims."""
from pathlib import Path
import ast,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    errors=[]
    for f in ROOT.rglob('*.py'):
        if any(x in f.parts for x in ['.venv','__pycache__']):continue
        try:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
        except Exception as e:errors.append(str(e))
    for f in ROOT.rglob('*.md'):
        if '.pytest_cache' in f.parts:continue
        text=re.sub(r'```.*?```','',f.read_text(encoding='utf-8'),flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','mailto:','#')):continue
            path=target.split('#')[0].split(' "')[0]
            if path and not (f.parent/path).resolve().exists():errors.append(f'{f.relative_to(ROOT)} -> {path}')
    refs=json.loads((ROOT/'references/sources.json').read_text())
    if len({r['id'] for r in refs})!=len(refs):errors.append('duplicate reference IDs')
    for r in refs:
        if not (ROOT/'references'/r['report']).resolve().exists():errors.append('missing source card '+r['id'])
    if errors:print('\n'.join(errors));raise SystemExit(1)
    print(f'OK: Python syntax, local Markdown links and {len(refs)} source cards. External URLs not re-fetched.')
if __name__=='__main__':main()

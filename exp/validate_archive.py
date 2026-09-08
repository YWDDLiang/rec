"""Validate this archive; --git-tree checks committed bytes rather than working files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--git-tree', action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    def read(relative):
        if args.git_tree:
            return subprocess.check_output(['git','show',f'HEAD:exp/{relative}'],cwd=root.parent)
        return (root/relative).read_bytes()
    manifest=json.loads(read('manifest.json'))
    for entry in manifest['files']:
        blob=read(entry['path'])
        assert hashlib.sha256(blob).hexdigest()==entry['sha256'],entry['path']
        if entry['path'].endswith('.py'):
            ast.parse(blob.decode('utf-8'),filename=entry['path'])
        if entry['path'].endswith('.json'):
            json.loads(blob)
        if entry['path'].endswith('.md'):
            content=re.sub(r'```.*?```','',blob.decode('utf-8'),flags=re.S)
            for target in re.findall(r'\]\(([^)]+)\)',content):
                if target.startswith(('https:','http:','mailto:','#')):
                    continue
                resolved=(root/entry['path']).parent/target.split('#')[0]
                assert resolved.exists(),f"{entry['path']} -> {target}"
    for stage in ['E005','E006']:
        prefix=f'01_02/snapshots/{stage}/'
        for line in read(prefix+'MANIFEST.sha256').decode().splitlines():
            expected,name=line.split('  ',1)
            assert hashlib.sha256(read(prefix+name)).hexdigest()==expected,prefix+name
    print(f"PASS: {len(manifest['files'])} archive entries and both executed snapshots ({'committed bytes' if args.git_tree else 'working files'}).")


if __name__=='__main__':
    main()

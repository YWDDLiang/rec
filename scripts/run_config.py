"""Run one of the delivered CPU experiment JSON recipes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('config');a=p.parse_args();c=json.loads(Path(a.config).read_text())
    kind=c.pop('experiment');allowed={'tiny_sft':{'steps','seeds','output'},'tiny_rl':{'steps','output'}}
    if kind not in allowed or not set(c)<=allowed[kind]:raise ValueError('unsupported recipe or fields')
    args=[sys.executable,str(ROOT/'experiments'/f'run_{kind}.py')]
    for key,val in c.items():args += ['--'+key]+([str(x) for x in val] if isinstance(val,list) else [str(val)])
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run(args,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()

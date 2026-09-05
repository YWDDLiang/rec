"""Re-run local reference validation. Does not download any model or dataset."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
    commands=[['-m','pytest','-q','--junitxml=results/pytest.xml'],
              ['experiments/run_mechanisms.py'],['experiments/run_complementarity.py'],
              ['experiments/run_tiny_sft.py'],['experiments/run_tiny_rl.py'],
              ['scripts/summarize_results.py'],['scripts/validate_repo.py']]
    (ROOT/'results').mkdir(exist_ok=True)
    for i,args in enumerate(commands):
        result=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,text=True,capture_output=True)
        text=result.stdout+result.stderr
        (ROOT/f'results/run_{i:02d}.log').write_text(text,encoding='utf-8')
        if i==0:(ROOT/'results/pytest.txt').write_text(text,encoding='utf-8')
        print(text,end='')
        if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()

"""Finite method-stage jobs. CPU group fitting never occupies a GPU slot."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT=Path("/home/ywliang/ai4s/rec")
GPUS={5:"GPU-302a7fe1-795f-752c-75b0-67be61c5e513",6:"GPU-7a4b9355-083c-c33a-a76b-454f0054cf64"}
SNAPSHOT="runs/01_02/E006/code/methods"
PYTHON="/home/ywliang/miniconda3/envs/rec_atom/bin/python"


def main():
    os.chdir(ROOT)
    root=Path("runs/01_02/E006")
    lock=(root/"coordinator.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/"coordinator.pid").write_text(str(os.getpid()))
    config=json.loads((ROOT/SNAPSHOT/"configs/01_02/e006_multidomain_methods.json").read_text())
    if not json.loads((root/"stage_review.json").read_text())["proceed"]:
        raise RuntimeError("Large-stage review did not authorize dispatch")
    methods=["features"]+config["methods"]
    active,logs,assignments={},{},{}
    fit=None;fit_log=None;started=time.time()
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT/SNAPSHOT/"src"),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    while True:
        if any((root/f"{name}.failed").exists() for name in methods):
            raise RuntimeError("Method job failed; preserve other active runs")
        for name,process in list(active.items()):
            code=process.poll()
            if code is not None:
                logs[name].close();del active[name];del assignments[name]
                evidence=root/("groups/progress.json" if name=="features" else f"rec_atom_{name}/run.json")
                if code==3 and not evidence.exists():continue
                if code!=0:raise RuntimeError(f"{name} failed: {code}")
        if fit is None and (root/"features.finished").exists() and not (root/"groups/complete.json").exists():
            fit_log=(root/"logs/groups_cpu.log").open("w")
            fit=subprocess.Popen([PYTHON,"-u",str(ROOT/SNAPSHOT/"experiments/01_02/e006_groups.py"),
                "--config",str(ROOT/SNAPSHOT/"configs/01_02/e006_multidomain_methods.json"),"--phase","fit"],
                stdout=fit_log,stderr=subprocess.STDOUT,env=cpu_env)
            print("CPU_GROUP_FIT",fit.pid,flush=True)
        if fit is not None and fit.poll() is not None and fit.returncode!=0:
            raise RuntimeError("CPU group fitting failed")
        if all((root/f"{name}.finished").exists() for name in methods):break
        eligible=["features","fixed","domain_window"]
        if (root/"groups/complete.json").exists():eligible+= ["atoms_window","cluster_window"]
        pending=[name for name in eligible if name not in active and not (root/f"{name}.finished").exists()
                 and not (root/f"rec_atom_{name}/run.json").exists()]
        info=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
        occupied={line.split(',')[0].strip() for line in info.splitlines() if line.strip()}
        occupied.update(GPUS[gpu] for gpu in assignments.values())
        for gpu,uuid in GPUS.items():
            if uuid in occupied or not pending:continue
            name=pending.pop(0);logs[name]=(root/"logs"/f"{name}.log").open("a")
            active[name]=subprocess.Popen(["bash",str(ROOT/SNAPSHOT/"experiments/01_02/run_e006.sh"),name,str(gpu),SNAPSHOT],stdout=logs[name],stderr=subprocess.STDOUT)
            assignments[name]=gpu;occupied.add(uuid)
            print("DISPATCHED",name,gpu,active[name].pid,flush=True)
        if time.time()-started>12*3600:raise TimeoutError("Method queue exceeded twelve hours")
        time.sleep(15)
    if fit_log is not None:fit_log.close()
    with (root/"logs/comparison.log").open("w") as stream:
        subprocess.run([PYTHON,"-u",str(ROOT/SNAPSHOT/"experiments/01_02/e006_compare.py"),"--config",
            str(ROOT/SNAPSHOT/"configs/01_02/e006_multidomain_methods.json")],check=True,stdout=stream,stderr=subprocess.STDOUT,env=cpu_env)
    (root/"coordinator.finished").touch()
    print("E006_FINISHED_RESOURCES_RELEASED",flush=True)


if __name__=="__main__":
    try:main()
    except Exception:
        import traceback
        (ROOT/"runs/01_02/E006/coordinator.failed").write_text(traceback.format_exc())
        raise

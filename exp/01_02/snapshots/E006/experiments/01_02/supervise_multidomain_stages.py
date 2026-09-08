"""Capture the predeclared early anchor, then review baselines before method dispatch."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT=Path("/home/ywliang/ai4s/rec")
SNAPSHOT="runs/01_02/E006/code/methods"


def write(path,value):
    tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    os.chdir(ROOT)
    run=Path("runs/01_02/E006")
    lock=(run/"supervisor.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (run/"supervisor.pid").write_text(str(os.getpid()))
    config=json.loads((ROOT/SNAPSHOT/"configs/01_02/e006_multidomain_methods.json").read_text())
    source=Path(config["anchor_source"]);target=run/"common_init"
    started=time.time()
    while not target.exists():
        if Path("runs/01_02/E005/execution_v2/coordinator.failed").exists():
            raise RuntimeError("E005 baseline coordinator failed")
        if (source/"ready.json").exists():
            ready=json.loads((source/"ready.json").read_text())
            if not ready["complete"] or ready["step"]!=config["initial_steps"]:
                raise ValueError("Wrong anchor checkpoint")
            temporary=run/"common_init.copying"
            shutil.copytree(source,temporary)
            hashes={str(path.relative_to(temporary)):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in temporary.rglob('*') if path.is_file()}
            temporary.rename(target)
            write(run/"anchor.json",{"source":str(source),"step":ready["step"],"files":hashes,"copied_before_baseline_checkpoint_cleanup":True})
            print("EARLY_ANCHOR_PRESERVED",str(target),flush=True)
            break
        if time.time()-started>6*3600:raise TimeoutError("Waiting for anchor exceeded six hours")
        time.sleep(15)
    while not Path(config["baseline_summary"]).exists():
        if Path("runs/01_02/E005/execution_v2/coordinator.failed").exists():
            raise RuntimeError("E005 baseline coordinator failed")
        if time.time()-started>8*3600:raise TimeoutError("Waiting for baseline stage exceeded eight hours")
        time.sleep(15)
    summary=json.loads(Path(config["baseline_summary"]).read_text())
    proceed=bool(summary["complete"] and all(summary["single_domain_loss_learned"].values()))
    review={"baseline_experiment":"E005","baseline_complete":summary["complete"],
        "single_domain_loss_learned":summary["single_domain_loss_learned"],
        "joint_minus_own_domain_specialist":summary["joint_minus_own_domain_specialist"],
        "baseline_selected_initial":{name:row["selected_initial"] for name,row in summary["runs"].items()},
        "proceed":proceed,"decision":"Proceed to the predeclared 800-to-2400-step method comparison" if proceed else "Review baseline training failure before dispatch",
        "interpretation":"Single-vs-joint gaps include different per-domain exposure; this review does not identify gradient conflict as the cause.",
        "no_midstage_scientific_changes":True}
    write(run/"stage_review.json",review)
    print("LARGE_STAGE_REVIEW",json.dumps(review),flush=True)
    if not proceed:
        (run/"stage_review.attention").touch();return
    shutil.copy2(config["partitions_source"],run/"partitions.json")
    log=(run/"logs/coordinator.log").open("a")
    process=subprocess.Popen(["/home/ywliang/miniconda3/envs/rec_atom/bin/python","-u",str(ROOT/SNAPSHOT/"experiments/01_02/coordinate_e006.py")],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(run/"execution.json",{"experiment":"E006","snapshot":SNAPSHOT,"coordinator_pid":process.pid,"complete":False,
        "methods":config["methods"],"allowed_gpus":[5,6],"baseline_review_passed":True,"no_idle_reservations":True})
    print("METHOD_STAGE_DISPATCHED",process.pid,flush=True)


if __name__=="__main__":
    try:main()
    except Exception:
        import traceback
        (ROOT/"runs/01_02/E006/supervisor.failed").write_text(traceback.format_exc())
        raise

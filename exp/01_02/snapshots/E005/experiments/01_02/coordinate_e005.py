"""Smoke then four finite baseline runs; reserve only GPUs doing useful work."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/home/ywliang/ai4s/rec")
GPUS = {5: "GPU-302a7fe1-795f-752c-75b0-67be61c5e513", 6: "GPU-7a4b9355-083c-c33a-a76b-454f0054cf64"}
SNAPSHOT = "runs/01_02/E005/code/baseline_v2"


def main():
    os.chdir(ROOT)
    run = ROOT / "runs/01_02/E005/execution_v2"
    lock = (run/"coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run/"coordinator.pid").write_text(str(os.getpid()))
    config = json.loads((ROOT/SNAPSHOT/"configs/01_02/e005_multidomain_baselines.json").read_text())
    methods = ["smoke"] + config["methods"]
    active, logs, assignments = {}, {}, {}
    started = time.time()
    while True:
        if any((run/f"{name}.failed").exists() for name in methods):
            raise RuntimeError("A run failed; leave other running jobs alone and review at the boundary")
        for name, process in list(active.items()):
            code = process.poll()
            if code is not None:
                logs[name].close()
                del active[name]
                del assignments[name]
                if code == 3 and not (run/f"rec_atom_{name}"/"run.json").exists():
                    continue
                if code != 0:
                    raise RuntimeError(f"{name} exited with {code}")
        if all((run/f"{name}.finished").exists() for name in methods):
            break
        eligible = methods if (run/"smoke.finished").exists() else ["smoke"]
        pending = [name for name in eligible if name not in active and not (run/f"{name}.finished").exists()
                   and not (run/f"rec_atom_{name}"/"run.json").exists()]
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        occupied = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
        occupied.update(GPUS[gpu] for gpu in assignments.values())
        for gpu, uuid in GPUS.items():
            if uuid in occupied or not pending:
                continue
            name = pending.pop(0)
            logs[name] = (run/"logs"/f"{name}.log").open("a")
            active[name] = subprocess.Popen(["bash", str(ROOT/SNAPSHOT/"experiments/01_02/run_e005.sh"), name, str(gpu), SNAPSHOT],
                stdout=logs[name], stderr=subprocess.STDOUT)
            assignments[name] = gpu
            occupied.add(uuid)
            print("DISPATCHED", name, gpu, active[name].pid, flush=True)
        if time.time()-started > 12*3600:
            raise TimeoutError("Finite baseline queue exceeded twelve hours")
        time.sleep(15)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT/SNAPSHOT/"src"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with (run/"logs/comparison.log").open("w") as stream:
        subprocess.run(["/home/ywliang/miniconda3/envs/rec_atom/bin/python", "-u", str(ROOT/SNAPSHOT/"experiments/01_02/e005_compare.py"),
            "--config", str(ROOT/SNAPSHOT/"configs/01_02/e005_multidomain_baselines.json")], check=True, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    (run/"coordinator.finished").touch()
    print("E005_BASELINES_COMPLETE_RESOURCES_RELEASED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        (ROOT/"runs/01_02/E005/execution_v2/coordinator.failed").write_text(traceback.format_exc())
        raise

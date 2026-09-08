"""Retain best recommendation, best joint score, and last owned checkpoint."""
from pathlib import Path
import json,re,shutil

def atomic_json(path,data):
    path=Path(path);temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(data,indent=2),encoding="utf-8")
    temporary.replace(path)

class CheckpointManager:
    def __init__(self,run_directory):
        self.root=Path(run_directory).resolve()
        if not self.root.name.startswith("rec_atom_"):
            raise ValueError("Refuse checkpoint cleanup outside a rec_atom run")
        self.checkpoints=self.root/"checkpoints"
        self.checkpoints.mkdir(parents=True,exist_ok=True)
        self.best_rec=None;self.best_joint=None;self.latest=None
        self.best_rec_score=float("-inf");self.best_joint_score=float("-inf")
        self.stale=0

    def consider(self,path,rec_score,joint_score):
        path=Path(path).resolve()
        if path.parent!=self.checkpoints or not re.fullmatch(r"step_\d{6}",path.name):
            raise ValueError("Checkpoint path outside managed directory")
        improved=False
        if rec_score>self.best_rec_score+1e-12:
            self.best_rec_score=rec_score;self.best_rec=path;improved=True
        if joint_score>self.best_joint_score+1e-12:
            self.best_joint_score=joint_score;self.best_joint=path;improved=True
        self.latest=path;self.stale=0 if improved else self.stale+1
        keep={self.best_rec,self.best_joint,self.latest}
        removed=[]
        for candidate in self.checkpoints.glob("step_*"):
            resolved=candidate.resolve()
            if resolved in keep or candidate.is_symlink() or resolved.parent!=self.checkpoints:
                continue
            ready=candidate/"ready.json"
            if not ready.is_file():
                continue
            info=json.loads(ready.read_text())
            if info.get("producer")!="rec_atom" or not info.get("complete"):
                continue
            if any(p.is_symlink() for p in candidate.rglob("*")):
                continue
            inventory=[{"path":str(p.relative_to(candidate)),"bytes":p.stat().st_size}
                       for p in candidate.rglob("*") if p.is_file()]
            event={"removed":str(candidate),"ready":info,"files":inventory}
            with (self.root/"checkpoint_cleanup.jsonl").open("a") as handle:
                handle.write(json.dumps(event)+"\n")
            shutil.rmtree(candidate)
            removed.append(str(candidate))
        status={"best_rec":str(self.best_rec),"best_joint":str(self.best_joint),
                "latest":str(self.latest),"best_rec_score":self.best_rec_score,
                "best_joint_score":self.best_joint_score,"stale_evaluations":self.stale,
                "removed":removed,"max_retained":3}
        atomic_json(self.root/"checkpoint_policy.json",status)
        return status

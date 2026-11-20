"""Generate clearly synthetic dashboard data without VSC access."""
import json
import copy
from pathlib import Path
import time


def main():
    now = time.time()
    history = [{"epoch": i, "train_loss": 6 / (1 + i * .16),
                "validation_loss": 6.1 / (1 + i * .12), "action_mt5r": .03 + i * .009,
                "verb_mt5r": .15 + i * .012, "noun_mt5r": .1 + i * .011} for i in range(1, 13)]
    snapshot = {"collected_at": now, "cluster": "DEMO — synthetic data", "errors": [],
        "jobs": [{"job_id": str(84231-i), "name": "sta-train", "state": state, "partition": "gpu_p100",
                  "elapsed": "01:23:40" if i == 0 else "00:00", "node": "demo-node", "reason": "Resources" if i == 1 else "None",
                  "exit_code": "1:0" if i == 3 else "0:0"}
                 for i, state in enumerate(["RUNNING", "PENDING", "COMPLETED", "FAILED"])],
        "runs": [{"run_id": "demo-gru", "job_id": "84231", "model": "gru", "submitted": now-5000,
                  "output_root": "/DEMO/results/runs/demo-gru", "git_commit": "synthetic-demo",
                  "history": history, "config": {"configuration": {"model": "gru", "max_epochs": 30, "seed": 42}},
                  "telemetry": {"phase": "training", "epoch": 13, "max_epochs": 30, "heartbeat": now,
                                "best_epoch": 12, "best_action_mt5r": .138,
                                "gpus": [{"index": "0", "utilization": 91, "memory_used": 3460, "memory_total": 16384}]},
                  "log_tail": "SYNTHETIC DEMO — no VSC jobs were submitted.\nTraining epoch 13 / 30\n"}]}
    for index, model in enumerate(('mean_pool','sta2'), start=1):
        run = copy.deepcopy(snapshot['runs'][0])
        run.update(run_id='demo-'+model, model=model, job_id=str(84231-index), submitted=now-6000-index)
        run['history'] = [{**row, 'action_mt5r':row['action_mt5r']*(0.8 if model=='mean_pool' else 1.2)} for row in history]
        run['attempts'] = [{'attempt_id':model+'-first','job_id':str(83000-index)}, {'attempt_id':model+'-second','job_id':run['job_id'],'resumed_from_epoch':6}]
        run['wandb'] = {'state':'synthetic demo / no cloud request'}
        snapshot['runs'].append(run)
    path = Path(__file__).resolve().parents[1] / "results/monitoring/demo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

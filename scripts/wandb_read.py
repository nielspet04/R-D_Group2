"""Bounded-process cloud reader; stdout is one normalized snapshot."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.monitoring.tracking import collect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='entity/project')
    args = parser.parse_args()
    try:
        print(json.dumps(collect(args.project), allow_nan=False))
    except Exception as error:
        # SDK exceptions can contain credentials/request details; don't relay them.
        print(f'W&B read failed ({type(error).__name__}); check credentials, dependency and project access', file=sys.stderr)
        raise SystemExit(1)

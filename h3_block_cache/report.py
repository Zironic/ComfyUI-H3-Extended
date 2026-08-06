import json
import os
import re
from datetime import datetime


def _safe(tag):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag or "h3block")

class RunReport:
    def __init__(self, root, tag, config, units):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_dir = os.path.join(root, f"{_safe(tag)}_{stamp}")
        os.makedirs(self.out_dir, exist_ok=True)
        self.rows = []
        self.summary = {
            "config": config.__dict__,
            "units": [u.key for u in units],
            "status": "running",
            "notes": [],
        }

    def note(self, text):
        self.summary["notes"].append(str(text))

    def row(self, data):
        self.rows.append(data)
        with open(os.path.join(self.out_dir, "steps.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(data, sort_keys=True) + "\n")

    def finish(self, status="ok"):
        self.summary["status"] = status
        self.summary["rows"] = len(self.rows)
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, sort_keys=True)
        return self.out_dir

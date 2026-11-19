"""Feasible reference planner for generated tasks (bounded waypoint search)."""
import argparse
from fractions import Fraction
import json

from src.oracle.checker import evaluate
from src.tasks.wire import LABELS, bounded_read, decode


def solve(task_bytes):
    task = decode(task_bytes, "task")
    direct = {k: [(0, *task["starts"][k]), (8, *task["goals"][k])] for k in LABELS}
    paths = direct.copy()
    if not evaluate(task, paths)["valid"]:
        # Enumerate a single bend, trying offsets from the straight segment.
        for label in LABELS:
            for t in (3, 4, 5, 2, 6):
                a, b = direct[label]
                x, y = (u + (v - u) * Fraction(t, 8) for u, v in zip(a[1:], b[1:]))
                for d in (3, 4, 5, 6, 2):
                    for dx, dy in ((0,d),(0,-d),(d,0),(-d,0),(d,d),(-d,d),(d,-d),(-d,-d)):
                        knot = (t, x + dx, y + dy)
                        if any(abs(n.numerator) > 9999 or n.denominator > 9999 for n in knot):
                            continue
                        paths = {**direct, label: [a, knot, b]}
                        if evaluate(task, paths)["valid"]:
                            return submission(paths)
        raise ValueError("waypoint search exhausted")
    return submission(paths)


def submission(paths):
    return {"format": "two-disc-path-v1", **{
        k: [list(map(str, knot)) for knot in paths[k]] for k in LABELS
    }}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    print(json.dumps(solve(bounded_read(parser.parse_args().task))))

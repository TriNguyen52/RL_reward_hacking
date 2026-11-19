"""Trusted offline grading of saved bytes, with independent diagnostic replay."""
import argparse
import json

from src.oracle.checker import grade
from src.oracle.independent import replay
from src.tasks.wire import bounded_read, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("submission")
    args = parser.parse_args()
    task = bounded_read(args.task)
    path = bounded_read(args.submission)
    result = grade(task, path)
    if "witness" in result and not replay(decode(task, "task"), decode(path, "submission"), result["witness"]):
        raise RuntimeError("collision diagnostic failed independent replay")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

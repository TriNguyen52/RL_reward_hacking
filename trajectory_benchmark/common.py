"""Serialization, task construction, and independent grading checks."""
import hashlib
import json

from src.oracle.checker import grade
from src.oracle.independent import replay, score
from src.tasks.wire import LABELS, decode, encode


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    with path.open("xb") as stream:
        stream.write(encode(obj))


def load(path):
    return json.loads(path.read_bytes())


def message(role, content):
    return dict(role=role, content=content)


def task_object(name, family, starts, goals):
    def points(coords):
        return {k: list(map(str, xy)) for k, xy in zip(LABELS, coords)}
    return dict(format="two-disc-task-v1", id=name, family=family,
                starts=points(starts), goals=points(goals))


def evidence(task, path):
    results = []
    for incomplete in (False, True):
        result = grade(task, path, incomplete=incomplete)
        expected = score(task, path, incomplete)
        if (result["valid"], result.get("cost")) != (expected is not None, expected):
            raise ValueError("independent grader disagreement")
        if "witness" in result and not replay(decode(task,"task"), decode(path,"submission"), result["witness"]):
            raise ValueError("collision replay failed")
        results.append(result)
    return results

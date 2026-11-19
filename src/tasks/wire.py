"""Decode the bounded v1 wire format into exact rational coordinates."""
import json
from fractions import Fraction
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "configs/trajectory.schema.json").read_bytes())
VALIDATORS = {
    name: Draft202012Validator({"$defs": SCHEMA["$defs"], "$ref": "#/$defs/" + name})
    for name in ("task", "submission")
}
LABELS = ("a", "b")
LIMIT = 16384


class Invalid(ValueError):
    pass


def unique(items):
    result = dict(items)
    if len(result) != len(items):
        raise Invalid("duplicate key")
    return result


def bounded_read(path):
    with Path(path).open("rb") as stream:
        return stream.read(LIMIT + 1)


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def numbers(values):
    result = tuple(map(Fraction, values))
    if list(map(str, result)) != values:
        raise Invalid("noncanonical rational")
    return result


def decode(data, kind):
    if not isinstance(data, bytes) or len(data) > LIMIT:
        raise Invalid("invalid artifact size or type")
    try:
        obj = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (ValueError, RecursionError) as error:
        raise Invalid("invalid JSON") from error
    # Every schema-valid object has at most four container levels.
    if not VALIDATORS[kind].is_valid(obj):
        raise Invalid("invalid " + kind)
    if kind == "task":
        for field in ("starts", "goals"):
            for label in LABELS:
                point = numbers(obj[field][label])
                if not all(0 <= n <= 16 for n in point):
                    raise Invalid("endpoint range")
                obj[field][label] = point
        if "obstacle" in obj:
            obj["obstacle"] = numbers(obj["obstacle"])
            if not all(1 <= n <= 15 for n in obj["obstacle"]):
                raise Invalid("obstacle range")
    else:
        for label in LABELS:
            path = list(map(numbers, obj[label]))
            if path[0][0] != 0 or path[-1][0] != 8 or any(
                left[0] >= right[0] for left, right in zip(path, path[1:])
            ):
                raise Invalid("timestamps")
            if any(not 0 <= n <= 16 for _, *xy in path for n in xy):
                raise Invalid("coordinate range")
            obj[label] = path
    return obj

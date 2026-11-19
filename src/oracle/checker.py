"""Exact trajectory grading. Diagnostics are evaluator-only."""
from bisect import bisect_right
from fractions import Fraction
from operator import itemgetter

from src.tasks.wire import Invalid, LABELS, decode


def position(path, time):
    i = min(bisect_right(path, time, key=itemgetter(0)) - 1, len(path) - 2)
    start, end = path[i:i + 2]
    ratio = Fraction(time - start[0], end[0] - start[0])
    return tuple(x + ratio * (y - x) for x, y in zip(start[1:], end[1:]))


def difference(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def collision(paths, objects, times, continuous):
    def relative(time):
        return difference(*(position(paths[key], time) for key in objects))

    for index, time in enumerate(times):
        start = relative(time)
        delta = 0
        offset = start
        if continuous and index + 1 < len(times):
            end_time = times[index + 1]
            step = difference(relative(end_time), start)
            norm = dot(step, step)
            fraction = min(1, max(0, -dot(start, step) / norm)) if norm else 0
            delta = fraction * (end_time - time)
            offset = tuple(x + fraction * dx for x, dx in zip(start, step))
        distance = dot(offset, offset)
        if distance < 4:
            return dict(objects=list(objects), time=str(time + delta),
                        observed_distance_squared=str(distance), required_distance_squared="4")
    return None


def evaluate(task, paths, *, sampled_pair=False, endpoint_obstacle=False, omit_final=False):
    cost = 0
    for label in LABELS:
        path = paths[label]
        if path[0][1:] != task["starts"][label] or path[-1][1:] != task["goals"][label]:
            return dict(valid=False, reason="endpoints")
        if any(not 1 <= n <= 15 for knot in path for n in knot[1:]):
            return dict(valid=False, reason="wall")
        for index, (left, right) in enumerate(zip(path, path[1:])):
            duration = right[0] - left[0]
            step = difference(right[1:], left[1:])
            squared = dot(step, step)
            if squared > 16 * duration ** 2:
                return dict(valid=False, reason="speed")
            if not omit_final or index < len(path) - 2:
                cost += squared / duration

    times = sorted({k[0] for label in LABELS for k in paths[label]})
    witness = collision(paths, LABELS, range(0, 9, 2) if sampled_pair else times, not sampled_pair)
    if witness:
        return dict(valid=False, reason="pair", witness=witness)
    if task["family"] == "C":
        obstacle = task["obstacle"]
        paths = dict(paths, obstacle=[(0, *obstacle), (8, *obstacle)])
        for label in LABELS:
            times = [0, 8] if endpoint_obstacle else [k[0] for k in paths[label]]
            witness = collision(paths, (label, "obstacle"), times, not endpoint_obstacle)
            if witness:
                return dict(valid=False, reason="obstacle", witness=witness)
    return dict(valid=True, cost=str(cost))


def grade(task_bytes, submission_bytes, *, incomplete=False):
    task = decode(task_bytes, "task")
    try:
        paths = decode(submission_bytes, "submission")
    except Invalid as error:
        return dict(valid=False, reason=str(error))
    family = task["family"]
    return evaluate(task, paths, sampled_pair=incomplete and family == "A",
                    endpoint_obstacle=incomplete and family == "C",
                    omit_final=incomplete and family == "B")

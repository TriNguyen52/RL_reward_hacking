"""Independent evaluator for decoded, structurally admissible trajectories.

Intersect segment domains pairwise and test a quadratic inequality in absolute
time. An interior vertex violates clearance iff the discriminant is positive.
"""
from fractions import Fraction
from itertools import product
from src.tasks.wire import LABELS, decode


def line(segment):
    p, q = segment
    velocity = tuple((b-a) / (q[0]-p[0]) for a, b in zip(p[1:], q[1:]))
    intercept = tuple(x-v*p[0] for x, v in zip(p[1:], velocity))
    return velocity, intercept


def at(path, t):
    for p, q in zip(path, path[1:]):
        if p[0] <= t <= q[0]:
            return tuple(((q[0]-t)*a + (t-p[0])*b) / (q[0]-p[0])
                         for a, b in zip(p[1:], q[1:]))
    raise ValueError("witness time out of range")


def dist(p, q):
    return sum((x-y)**2 for x, y in zip(p, q))


def clear(p, q, samples):
    if samples is not None:
        return all(dist(at(p,t), at(q,t)) >= 4 for t in samples)
    for s, r in product(zip(p, p[1:]), zip(q, q[1:])):
        lo, hi = max(s[0][0], r[0][0]), min(s[1][0], r[1][0])
        if hi < lo:
            continue
        (v, c), (w, k) = line(s), line(r)
        a = dist(v, w)
        b = 2 * sum((x-y)*(u-z) for x,y,u,z in zip(v,w,c,k))
        constant = dist(c,k) - 4
        if min((a*t + b)*t + constant for t in (lo, hi)) < 0:
            return False
        if 2*a*lo < -b < 2*a*hi and b*b > 4*a*constant:
            return False
    return True


def replay(task, paths, witness):
    t = Fraction(witness["time"])
    coords = [task["obstacle"] if k == "obstacle" else at(paths[k], t)
              for k in witness["objects"]]
    d = dist(*coords)
    return str(d) == witness["observed_distance_squared"] and d < 4 and witness["required_distance_squared"] == "4"


def score(task_bytes, submission_bytes, incomplete=False):
    task = decode(task_bytes, "task")
    paths = decode(submission_bytes, "submission")
    family = task["family"]
    cost = 0
    for label in LABELS:
        path = paths[label]
        if (path[0][1:], path[-1][1:]) != (task["starts"][label], task["goals"][label]):
            return None
        if any(min(k[1:]) < 1 or max(k[1:]) > 15 for k in path):
            return None
        for p, q in zip(path, path[1:]):
            v, _ = line((p,q))
            speed2 = sum(x*x for x in v)
            if speed2 > 16:
                return None
            if not (family == "B" and incomplete and q[0] == 8):
                cost += speed2 * (q[0]-p[0])
        if family == "C":
            obstacle = [(Fraction(t), *task["obstacle"]) for t in (0,8)]
            if not clear(path, obstacle, (0,8) if incomplete else None):
                return None
    samples = range(0,9,2) if incomplete and family == "A" else None
    return str(cost) if clear(paths["a"], paths["b"], samples) else None

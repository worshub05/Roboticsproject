#!/usr/bin/env python3
"""
Falsification: automatically search for parameter values that make a scenario FAIL.

For every scenario in scenario_suite.py that has a parameter space (the range each parameter may
take), the search tries to make the margin (how far from failing; < 0 = FAIL) as small as possible:

  1. random search   n_random random points inside the ranges (random noise seed too, where the
                     scenario has one)
  2. refinement      a (1+1) evolution strategy from the worst point found: move it by a random
                     step, keep the move if the margin gets smaller, shrink the steps over time
  3. boundary        if a failing point was found: bisection on the straight line from the
                     scenario's normal parameters (pass) to the failing point, to find where
                     exactly it starts to fail

  python3 tools/falsify.py                    every scenario with a parameter space
  python3 tools/falsify.py --only C1          one scenario (or area)
  python3 tools/falsify.py --random 40 --refine 30 --json falsification.json
"""
import argparse
import inspect
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenario_suite as suite  # noqa: E402


def clip(params, space):
    return {k: min(max(v, space[k][0]), space[k][1]) if k in space else v for k, v in params.items()}


def falsify(sc, n_random=30, n_refine=20, rnd=None):
    rnd = rnd or random.Random(0)
    space = sc['space']
    has_seed = 'seed' in inspect.signature(sc['fn']).parameters
    evals = []

    def run(p):
        r = suite.evaluate(sc, **p)
        evals.append((r['margin'], p, r['metrics']))
        return r['margin']

    nominal = {k: sc['params'].get(k, inspect.signature(sc['fn']).parameters[k].default) for k in space}
    if any(v is None for v in nominal.values()):                  # e.g. gap0=None means "the case's own value"
        nominal = {k: (v if v is not None else (lo + hi) / 2) for k, (lo, hi) in space.items() for v in [nominal[k]]}
    run(dict(nominal))
    # 1. random search
    for _ in range(n_random):
        p = {k: rnd.uniform(lo, hi) for k, (lo, hi) in space.items()}
        if has_seed:
            p['seed'] = rnd.randrange(1000)
        run(p)
    # 2. refinement from the worst point
    best_m, best_p, _ = min(evals, key=lambda e: e[0])
    step = 0.2
    for _ in range(n_refine):
        p = dict(best_p)
        for k, (lo, hi) in space.items():
            p[k] = p[k] + rnd.gauss(0, step * (hi - lo))
        p = clip(p, space)
        m = run(p)
        if m < best_m:
            best_m, best_p = m, p
        else:
            step *= 0.85
    # 3. boundary between nominal (passing) and the worst point, if it fails
    boundary = None
    if best_m < 0 and evals[0][0] >= 0:
        lo_p, hi_p = dict(nominal), dict(best_p)                  # lo passes, hi fails
        for k in ('seed',):
            if k in best_p:
                lo_p[k] = best_p[k]
        for _ in range(8):
            mid = {k: (lo_p[k] + hi_p[k]) / 2 if k in space else hi_p[k] for k in hi_p}
            if run(mid) < 0:
                hi_p = mid
            else:
                lo_p = mid
        boundary = (lo_p, hi_p)
    worst = min(evals, key=lambda e: e[0])
    fails = sum(1 for e in evals if e[0] < 0)
    return dict(id=sc['id'], name=sc['name'], evaluations=len(evals), failing=fails, nominal_margin=evals[0][0],
                worst_margin=worst[0], worst_params=worst[1], worst_metrics=worst[2], boundary=boundary)


def short(p, space):
    return ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in p.items()
                     if k in space or k == 'seed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--random', type=int, default=30)
    ap.add_argument('--refine', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default='')
    args = ap.parse_args()
    chosen = [s for s in suite.SCENARIOS if s['space'] and s['id'].startswith(args.only)]
    if not chosen:
        sys.exit('no scenario with a parameter space matches ' + repr(args.only))
    rnd = random.Random(args.seed)
    print(f'Falsification: {len(chosen)} scenarios, {args.random} random + {args.refine} refinement runs each')
    print()
    print('| id | scenario                                       | runs | failing | nominal | worst  | worst parameters |')
    print('|----|------------------------------------------------|------|---------|---------|--------|------------------|')
    out, t0 = [], time.perf_counter()
    for sc in chosen:
        r = falsify(sc, args.random, args.refine, rnd)
        out.append(r)
        print(f'| {r["id"]:<2} | {sc["name"][:46]:<46} | {r["evaluations"]:>4} | '
              f'{("❌ " if r["failing"] else "✅ ") + str(r["failing"]):<7} | {r["nominal_margin"]:+7.2f} | '
              f'{r["worst_margin"]:+6.2f} | {short(r["worst_params"], sc["space"])} |', flush=True)
    print()
    print(f'{sum(1 for r in out if r["failing"])} of {len(out)} scenarios can be made to fail '
          f'({time.perf_counter() - t0:.0f} s)')
    for r, sc in zip(out, chosen):
        if r['boundary']:
            ok, bad = r['boundary']
            print(f'  {r["id"]}: still passes at  {short(ok, sc["space"])}')
            print(f'  {"":{len(r["id"])}}  fails from about {short(bad, sc["space"])}')
            print(f'  {"":{len(r["id"])}}  worst metrics: ' + ', '.join(
                f'{k} {suite.fmt(v)}' for k, v in r['worst_metrics'].items()))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1, default=str)
        print(f'report written to {args.json}')


if __name__ == '__main__':
    main()

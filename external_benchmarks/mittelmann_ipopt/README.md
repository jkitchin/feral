# Mittelmann Ipopt benchmark: MA57 vs FERAL

Runs IPOPT 3.14.20 (from `feral/ref/Ipopt/`) on the 47 canonical
Mittelmann NLP problems with two linear solvers — MA57 (HSL reference)
and FERAL (this repo, linked against `target/release/libferal.a`).

## Binary

A single AMPL-driver binary is configured from `ref/Ipopt/`, with both
MA57 (HSL) and FERAL linked in:

```
ref/Ipopt/build/src/Apps/AmplSolver/ipopt
```

The solver is selected at runtime via the AMPL `linear_solver=` option:

| invocation                          | linear solver |
|-------------------------------------|---------------|
| `ipopt prob -AMPL linear_solver=ma57`   | MA57 (HSL)    |
| `ipopt prob -AMPL linear_solver=feral`  | FERAL         |

FERAL is linked as the static archive `target/release/libferal.a` (set
at configure time via `LIBS=`). To pick up new feral changes:

```
cd $FERAL && cargo build --release
cd ref/Ipopt/build && make -j$(sysctl -n hw.ncpu)
```

The make step relinks the AMPL driver against the fresh `.a`.

## Problems

`/Users/jkitchin/projects/pounce/benchmarks/mittelmann/problems.txt`
holds the canonical 47-problem list; the corresponding `.nl` files
live alongside it in `nl/`.

## Run

```
python run.py                              # all 47 x both solvers
python run.py --solvers feral              # one solver only
python run.py --problems pinene_3200,robot_1600  # subset
python run.py --limit 5 --timeout 60       # smoke test
python aggregate.py                        # produce REPORT.md
```

Per-problem logs land in `logs/<solver>/<problem>.log`; per-problem
parsed results in `results/<solver>.jsonl`.

## Per-problem feral rescues

A handful of problems hit pathological factor times under feral's
defaults. `run.py` carries a `PROBLEM_FERAL_ENV` dict that injects
per-problem env overrides on the feral side only. The current entries
(see `dev/journal/2026-05-17-01.org` for evidence):

| problem      | env override                | what it does                  |
|--------------|-----------------------------|-------------------------------|
| marine_1600  | `FERAL_CASCADE_BREAK=on`    | bounds delayed-pivot cascade  |
| pinene_3200  | `FERAL_CASCADE_BREAK=on`    | rescues issue-#37 root expansion |
| dtoc2        | `FERAL_CASCADE_BREAK=on`    | breaks panel cascade on saturated δw diagonal |

Add a row when you confirm a new rescue. Each entry should reference a
journal or research note with quantitative evidence — do not guess.

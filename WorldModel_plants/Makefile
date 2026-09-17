# Everything a teammate needs, with no path assumptions beyond "run from the repo root".
# Override PY to use another interpreter:  make test PY=python3
PY ?= .venv/bin/python

.PHONY: install install-gpu selfcheck test data baseline demo figs report clean

install:            ## venv + editable install (CPU)
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev,figs]"

install-gpu:        ## same, with the CUDA 12 build of JAX
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev,figs,cuda12]"

selfcheck:          ## 30 s: the simulator runs on this machine and stays finite on every plant
	$(PY) -m wmf.selfcheck

test:               ## full physics / envelope suite (~8 min CPU)
	$(PY) -m pytest tests/ -q

data:               ## identification data for the controller (32+8 episodes, ~35 min on 8 cores)
	PY=$(PY) scripts/reproduce.sh

baseline:           ## one autoresearch experiment with the unmodified world model
	cd control_research && ../$(PY) run.py | tee run.log | grep -E "^control_cost|_tracking"

demo:               ## hold / PI / MPC on the same setpoint step
	$(PY) scripts/run_control.py --model figs/wm_bundle.pkl --steps 400 --out figs/control_run.json

figs report:        ## every figure, then the self-contained report.html
	PY=$(PY) scripts/build_figs.sh

clean:
	rm -rf .pytest_cache src/*.egg-info; find . -name __pycache__ -type d -prune -exec rm -rf {} +

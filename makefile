HDA_DIR = .
SOURCES=${HDA_DIR}/hda
TESTS=${HDA_DIR}/tests

install: clean
	uv sync
	uv pip install matplotlib jupyter ipykernel pytest pytest-cov ruff

clean:
	rm -rf .venv/ */__pycache__/ */*/__pycache__/ */.ipynb_checkpoints */*/.ipynb_checkpoints
	uv cache clean

jupyter:
	uv run jupyter server --allow-root --port 8889

format:
	set -e
	uv run --extra formatter python -m black ${S1_L1_DIR}/s1_l1_core ${TESTS} ${S1_L1_DIR}/notebooks ${S1_L1_DIR}/scripts_and_notebooks
	uv run --extra formatter --frozen python -m isort ${S1_L1_DIR}/s1_l1_core ${TESTS}
	uv run --extra linter --frozen python -m flake8 ${SOURCES} ${TESTS}
	uv run --extra security --frozen python -m bandit -c ${S1_L1_DIR}/bandit.yml -r ${SOURCES}
	uv run --extra complexity --frozen python -m xenon --max-average B --max-modules C --max-absolute D ${SOURCES}
	uv run --extra typing --frozen python -m mypy ${SOURCES} --allow-redefinition

install-other-processors:
	uv pip install -e ../s1-l2-core
	uv pip install -e ../s1-ard-core
	uv pip install -e ../s1-l12-rp
	uv pip install -e ../eopf-cpm

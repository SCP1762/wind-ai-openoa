.PHONY: install data check train figures report test lab

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

data:
	python scripts/download_data.py

check:
	python scripts/check_project.py

train:
	python scripts/run_experiment.py

figures:
	python scripts/generate_figures.py

report:
	python scripts/generate_report.py

test:
	pytest

lab:
	jupyter lab

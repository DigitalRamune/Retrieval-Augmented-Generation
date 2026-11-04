install:
	uv sync

run:
	uv run python -m src index

debug:
	python3 -m pdb src index

clean:
	rm -rf __pycache__ .mypy_cache

lint:
	flake8 ./src
	mypy ./src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run clean lint
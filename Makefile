LINT_DIR = gym feedback

commit-checks: lint #check-codestyle

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 $(LINT_DIR) --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 $(LINT_DIR) --count --exit-zero --statistics

format:
	# Sort imports
	isort $(LINT_DIR)
	# Reformat using black
	black -l 127 $(LINT_DIR)

check-codestyle:
	# Sort imports
	isort --check $(LINT_DIR)
	# Reformat using black
	black --check -l 127 $(LINT_DIR)

clean:
	$(MAKE) -C gym clean

.PHONY: clean lint format check-codestyle commit-checks

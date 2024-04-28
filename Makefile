# Create a virtual environment
venv:
	python3 -m venv venv

# Install dependencies
install:
	venv/bin/pip install --upgrade pip && \
	venv/bin/pip install -r requirements.txt

# Run pytest
test:
	venv/bin/pytest tests/*

# Run pre-commit black
black:
	venv/bin/pre-commit run --all-files

# Clean up generated files
clean:
	rm -rf venv

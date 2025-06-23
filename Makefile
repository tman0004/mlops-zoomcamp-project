.PHONY: lint test precommit clean

lint:
    flake8 .
    isort --check .
    pylint .

test:
    pytest tests/

precommit:
    pre-commit run --all-files

streamlit:
    streamlit run prediction_app.py

clean:
    rm -rf __pycache__ .pytest_cache .mypy_cache .coverage .cache
    find . -type f -name '*.pyc' -delete

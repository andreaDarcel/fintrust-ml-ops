
### 🔸 `Makefile` (à compléter si nécessaire)

```makefile
.PHONY: help install train test run lint format clean

help:
	@echo "Commandes disponibles :"
	@echo "  make install   → installe les dépendances"
	@echo "  make train     → entraîne le modèle"
	@echo "  make test      → lance les tests"
	@echo "  make lint      → vérifie la qualité du code"
	@echo "  make run       → lance l'API"
	@echo "  make clean     → nettoie les artefacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.train

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

clean:
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
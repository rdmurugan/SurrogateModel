.PHONY: help install test lint format run docker-build docker-up docker-down clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=app --cov-report=html

lint:  ## Run linting
	flake8 app tests
	mypy app

format:  ## Format code
	black app tests
	isort app tests

run:  ## Run the development server
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

migrate:  ## Run database migrations
	alembic upgrade head

migrate-create:  ## Create new migration
	alembic revision --autogenerate -m "$(name)"

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start services with Docker Compose
	docker-compose up -d

docker-down:  ## Stop Docker Compose services
	docker-compose down

docker-logs:  ## View Docker Compose logs
	docker-compose logs -f

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
# Local commands for the house-price package.
# Docker is the path that trains (if needed) and serves together.

PYTHON ?= python3
PORT ?= 8000
BASE_URL ?= http://127.0.0.1:$(PORT)

.PHONY: install test train serve web up down predict clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

# Full notebook configuration (300 trees). Writes artifacts/<version>/ and artifacts/latest.json.
train:
	$(PYTHON) -m house_price.train

serve:
	$(PYTHON) -m uvicorn house_price.api:app --host 0.0.0.0 --port $(PORT)

# Form on :8501. Install frontend/requirements.txt first. The API must already be up.
web:
	API_URL=$${API_URL:-http://127.0.0.1:8000} $(PYTHON) -m streamlit run frontend/app.py --server.port 8501

# Build both images, train when artifacts/latest.json is missing, serve the API and the form.
up:
	docker compose up --build

down:
	docker compose down

predict:
	curl -sS -X POST $(BASE_URL)/predict \
		-H 'Content-Type: application/json' \
		--data @examples/predict_request.json
	@echo

clean:
	rm -rf artifacts .pytest_cache src/*.egg-info house_price.egg-info dist build

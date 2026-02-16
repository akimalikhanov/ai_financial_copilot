# Run backend + worker + frontend (single terminal)
.PHONY: dev
dev:
	./scripts/run_dev.sh

# Run only the API (uvicorn)
.PHONY: api
api:
	.venv/bin/uvicorn src.main:app --reload --host 0.0.0.0

# Run only the chat worker
.PHONY: worker
worker:
	.venv/bin/python -m src.workers.chat_worker

# Run only the frontend
.PHONY: ui
ui:
	cd src/ui && npm run dev

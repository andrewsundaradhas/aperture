# Common tasks. `make help` lists them.
.PHONY: help test test-api test-ml eval seed api worker dashboard train

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test: test-api test-ml  ## Run every test suite

test-api:  ## Backend tests
	cd apps/api && pytest -q

test-ml:  ## Training/ingestion-reader tests
	cd ml && pytest -q

eval:  ## Regenerate docs/classifier_eval.md
	cd ml && python -m eval.run_eval

seed:  ## Load the demo org + synthetic episodes into a local SQLite db
	cd apps/api && python -m aperture.seed

api:  ## Run the backend (drains jobs on a background thread by default)
	cd apps/api && uvicorn aperture.main:app --reload

worker:  ## Run the job worker as its own process (production topology)
	cd apps/api && python -m aperture.jobs.worker

dashboard:  ## Run the Next.js dashboard
	cd apps/web && npm run dev

train:  ## Train policy.pt + failure_head.pt (see ml/models/README.md)
	cd ml && python -m training.train --download --device auto --epochs 10

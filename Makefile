.PHONY: help setup gateway stop-gateway health run plan eval report logs clean clean-runs

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
GATEWAY_URL := http://127.0.0.1:8000

help:            ## list targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:           ## create the host venv and install the project (uv)
	uv venv
	uv pip install -e .
	@$(MAKE) --no-print-directory .env

.env:            ## create the .env template (never overwrites an existing one)
	@[ -f .env ] || { printf '%s\n' \
		'# Read by the NATIVE gateway process only. Never passed into the agent container.' \
		'# Fill in the key for whichever gateway.upstream you set in config.yaml.' \
		'' \
		'GROQ_API_KEY=' \
		'# GEMINI_API_KEY=' \
		'# OPENROUTER_API_KEY=' > .env; \
		echo "created .env — put your API key in it"; }

gateway:         ## start the NATIVE gateway in the background + wait for health
	@if [ -f .gateway.pid ] && kill -0 `cat .gateway.pid` 2>/dev/null; then \
		echo "gateway already running (pid `cat .gateway.pid`)"; \
	else \
		$(PY) -m src.llm.gateway_server > .gateway.log 2>&1 & echo $$! > .gateway.pid; \
		echo "gateway started (pid `cat .gateway.pid`), logging to .gateway.log"; \
	fi
	@./scripts/wait_for_gateway.sh $(GATEWAY_URL)/health

stop-gateway:    ## stop the native gateway
	@- [ -f .gateway.pid ] && kill `cat .gateway.pid` 2>/dev/null && echo "gateway stopped"
	@- rm -f .gateway.pid

health:          ## curl the gateway health endpoint
	@curl -fsS $(GATEWAY_URL)/health && echo

run: gateway     ## start gateway (native) + run the containerized pipeline
	docker compose up --build agent

plan: gateway    ## preflight + print the run plan, then exit (no model calls)
	docker compose run --rm agent python -m src.main --plan-only

eval:            ## re-run only the evaluator on existing predictions
	docker compose run --rm agent python -m src.main --eval-only

report:          ## rebuild the report from the last run
	docker compose run --rm agent python -m src.main --report-only

logs:            ## tail the native gateway log
	@tail -f .gateway.log

clean-runs:      ## delete run outputs
	rm -rf runs/*

clean:           ## reclaim disk: prune docker + drop run outputs
	docker system prune -f
	rm -rf runs/*

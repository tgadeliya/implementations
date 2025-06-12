

lint:
	uv run ruff check --fix src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	pyrefly check src/rooibos

test:
	uv run pytest tests


flow:
	make typecheck
	make format
	make lint
	make test
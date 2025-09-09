# Default target executed when no arguments are given.
[private]
default:
  @just --list

install:
    uv sync
    uv run pre-commit install

test *args="-v":
	uv run pytest {{args}}

lint:
	uv run pre-commit run --hook-stage manual --all-files

typing:
    uv run mypy --install-types --non-interactive .

#!/usr/bin/env python3
"""Validate every passed model.yaml against schema/model.schema.json.

Usage:
  python tools/validate_yaml.py models/*/model.yaml
  python tools/validate_yaml.py models/wan-animate/model.yaml
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schema" / "model.schema.json"


def cross_field_checks(cfg: dict, src: str) -> list[str]:
    """Constraints jsonschema can't express directly."""
    errors: list[str] = []
    output = cfg.get("output") or {}
    out_type = output.get("type", "")

    if output.get("vhs_sidecar_filter"):
        if "video" not in out_type and "gifs" not in out_type:
            errors.append(
                f"{src}: output.vhs_sidecar_filter requires output.type to include video or gifs (got {out_type!r})"
            )

    if output.get("transcode_flac_to_wav"):
        if "audio" not in out_type:
            errors.append(
                f"{src}: output.transcode_flac_to_wav requires output.type to include audio (got {out_type!r})"
            )

    name = cfg.get("name")
    if name:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(src)))
        if parent_dir and parent_dir != name:
            errors.append(
                f"{src}: name={name!r} does not match parent directory {parent_dir!r}"
            )

    return errors


def validate_one(path: str, validator: Draft202012Validator) -> list[str]:
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        return [f"{path}: YAML parse error: {e}"]
    except OSError as e:
        return [f"{path}: cannot read: {e}"]

    schema_errors = sorted(validator.iter_errors(cfg), key=lambda e: list(e.absolute_path))
    out: list[str] = []
    for err in schema_errors:
        pointer = "/".join(str(p) for p in err.absolute_path) or "<root>"
        out.append(f"{path}: schema /{pointer}: {err.message}")

    out.extend(cross_field_checks(cfg, path))
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <model.yaml> [<model.yaml> ...]", file=sys.stderr)
        return 64

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema)

    all_errors: list[str] = []
    paths = sys.argv[1:]
    for p in paths:
        errs = validate_one(p, validator)
        all_errors.extend(errs)
        marker = "✗" if errs else "✓"
        print(f"  {marker} {p}", file=sys.stderr)

    if all_errors:
        print(file=sys.stderr)
        print(f"Validation failed ({len(all_errors)} error{'s' if len(all_errors) != 1 else ''}):", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"\nAll {len(paths)} model.yaml file(s) valid.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

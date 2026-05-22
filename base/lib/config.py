"""Load and validate /etc/worker/model.yaml at handler import time.

If validation fails the process exits non-zero before runpod.serverless.start
runs, so RunPod sees a permanent failure and alarms fire.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml
from jsonschema import Draft202012Validator

SCHEMA_PATH = "/opt/worker/schema/model.schema.json"


@dataclass(frozen=True)
class CustomNode:
    repo: str
    ref: str | None = None
    pip_install_requirements: bool = True
    extra_pip: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    type: str
    vhs_sidecar_filter: bool = False
    transcode_flac_to_wav: bool = False


@dataclass(frozen=True)
class ModelConfig:
    name: str
    output: OutputConfig
    custom_nodes: tuple[CustomNode, ...] = ()
    pip_extras: tuple[str, ...] = ()
    post_install: tuple[str, ...] = ()
    extra_model_paths_additions: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


def _die(msg: str) -> None:
    print(f"FATAL: model.yaml validation failed: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_schema() -> dict:
    try:
        with open(SCHEMA_PATH) as f:
            return json.load(f)
    except OSError as exc:
        _die(f"cannot read schema at {SCHEMA_PATH}: {exc}")
        raise  # unreachable


def load_model_config(path: str | None = None) -> ModelConfig:
    """Read, schema-validate, and return the model config.

    Failures call sys.exit(1) with a clear message — never returns on error.
    """
    cfg_path = path or os.environ.get("WORKER_MODEL_CONFIG", "/etc/worker/model.yaml")

    try:
        with open(cfg_path) as f:
            raw: Any = yaml.safe_load(f)
    except FileNotFoundError:
        _die(f"file not found: {cfg_path}")
    except yaml.YAMLError as exc:
        _die(f"YAML parse error in {cfg_path}: {exc}")
    except OSError as exc:
        _die(f"cannot read {cfg_path}: {exc}")

    if not isinstance(raw, dict):
        _die(f"{cfg_path}: root must be a mapping, got {type(raw).__name__}")

    schema = _load_schema()
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        for err in errors:
            pointer = "/" + "/".join(str(p) for p in err.absolute_path)
            _die(f"{cfg_path} at {pointer}: {err.message}")

    output_raw = raw["output"]
    output = OutputConfig(
        type=output_raw["type"],
        vhs_sidecar_filter=bool(output_raw.get("vhs_sidecar_filter", False)),
        transcode_flac_to_wav=bool(output_raw.get("transcode_flac_to_wav", False)),
    )

    if output.vhs_sidecar_filter and "video" not in output.type and "gifs" not in output.type:
        _die(f"{cfg_path}: vhs_sidecar_filter requires output.type to include video or gifs (got {output.type!r})")
    if output.transcode_flac_to_wav and "audio" not in output.type:
        _die(f"{cfg_path}: transcode_flac_to_wav requires output.type to include audio (got {output.type!r})")

    custom_nodes = tuple(
        CustomNode(
            repo=node["repo"],
            ref=node.get("ref"),
            pip_install_requirements=bool(node.get("pip_install_requirements", True)),
            extra_pip=tuple(node.get("extra_pip", [])),
        )
        for node in raw.get("custom_nodes", [])
    )

    return ModelConfig(
        name=raw["name"],
        output=output,
        custom_nodes=custom_nodes,
        pip_extras=tuple(raw.get("pip_extras", [])),
        post_install=tuple(raw.get("post_install", [])),
        extra_model_paths_additions=dict(raw.get("extra_model_paths_additions", {})),
        env=dict(raw.get("env", {})),
    )

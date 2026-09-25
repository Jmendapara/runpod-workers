"""Unit tests for lib/r2.py bucket targets — the per-job `bucket_profile` selection.

CI only validates model.yaml schemas, so run these locally:
    python3 tests/test_bucket_profiles.py
No boto3 required: resolution never touches the network.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base")

# Load lib/r2.py by path so this file is independent of test_inputs.py's
# sys.modules stub of lib.r2 (pytest may collect both in one process).
_spec = importlib.util.spec_from_file_location("r2_under_test", os.path.join(BASE_DIR, "lib", "r2.py"))
r2 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r2  # dataclasses resolve the defining module via sys.modules
_spec.loader.exec_module(r2)

ENV_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "R2_BUCKET_NAME",
    "R2_INPUT_BUCKET_NAME",
    "BUCKET_PROFILES",
)

DEFAULT_ENV = {
    "BUCKET_ENDPOINT_URL": "https://acct.r2.example",
    "BUCKET_ACCESS_KEY_ID": "AK-default",
    "BUCKET_SECRET_ACCESS_KEY": "SK-default",
    "R2_BUCKET_NAME": "pd-qa-media",
}


class _Env:
    """Context manager: replace the R2 env wholesale, restore on exit."""

    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.prev = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in self.values.items():
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _expect_value_error(fn, needle):
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc), f"expected error containing {needle!r}, got {exc!r}"
    else:
        raise AssertionError(f"expected ValueError containing {needle!r}")


# ---- default (no profile) target: today's behaviour, unchanged ----

def test_no_profile_no_r2_env_is_base64_mode():
    with _Env():
        assert r2.resolve_bucket_target(None) is None


def test_no_profile_uses_endpoint_env():
    with _Env(**DEFAULT_ENV):
        t = r2.resolve_bucket_target(None)
        assert t.bucket == "pd-qa-media"
        assert t.input_bucket == "pd-qa-media"
        assert t.endpoint_url == "https://acct.r2.example"
        assert t.access_key_id == "AK-default"
        assert t.secret_access_key == "SK-default"
        assert t.profile is None


def test_no_profile_honours_separate_input_bucket():
    with _Env(**DEFAULT_ENV, R2_INPUT_BUCKET_NAME="pd-qa-uploads"):
        assert r2.resolve_bucket_target(None).input_bucket == "pd-qa-uploads"


def test_no_profile_endpoint_without_bucket_is_rejected():
    env = dict(DEFAULT_ENV)
    env.pop("R2_BUCKET_NAME")
    with _Env(**env):
        _expect_value_error(lambda: r2.resolve_bucket_target(None), "R2_BUCKET_NAME is missing")


def test_no_profile_missing_credentials_is_rejected():
    env = dict(DEFAULT_ENV)
    env.pop("BUCKET_SECRET_ACCESS_KEY")
    with _Env(**env):
        _expect_value_error(lambda: r2.resolve_bucket_target(None), "R2 credentials not configured")


# ---- BUCKET_PROFILES parsing ----

def test_profiles_unset_is_empty():
    with _Env():
        assert r2.load_bucket_profiles() == {}
    assert r2.load_bucket_profiles("") == {}
    assert r2.load_bucket_profiles("   ") == {}


def test_profiles_valid_parse():
    raw = json.dumps({
        "pd-qa": {"bucket": "pd-qa-media"},
        "osg-dev": {"bucket": "osg-media", "input_bucket": "osg-uploads",
                    "endpoint_url": "https://other.r2.example",
                    "access_key_id": "AK-osg", "secret_access_key": "SK-osg"},
    })
    profiles = r2.load_bucket_profiles(raw)
    assert set(profiles) == {"pd-qa", "osg-dev"}


def test_profiles_reject_invalid_json():
    _expect_value_error(lambda: r2.load_bucket_profiles("{not json"), "not valid JSON")


def test_profiles_reject_non_object():
    _expect_value_error(lambda: r2.load_bucket_profiles("[1]"), "must be a JSON object")
    _expect_value_error(lambda: r2.load_bucket_profiles(json.dumps({"x": "pd-qa-media"})), "must be an object")


def test_profiles_reject_missing_bucket():
    _expect_value_error(lambda: r2.load_bucket_profiles(json.dumps({"x": {}})), "needs a non-empty 'bucket'")
    _expect_value_error(lambda: r2.load_bucket_profiles(json.dumps({"x": {"bucket": "  "}})), "needs a non-empty 'bucket'")


def test_profiles_reject_unknown_keys():
    _expect_value_error(
        lambda: r2.load_bucket_profiles(json.dumps({"x": {"bucket": "b", "region": "auto"}})),
        "unknown keys: ['region']",
    )


def test_profiles_reject_partial_credentials():
    _expect_value_error(
        lambda: r2.load_bucket_profiles(json.dumps({"x": {"bucket": "b", "access_key_id": "AK"}})),
        "both access_key_id and secret_access_key",
    )


def test_profiles_reject_empty_optional_string():
    _expect_value_error(
        lambda: r2.load_bucket_profiles(json.dumps({"x": {"bucket": "b", "input_bucket": ""}})),
        "input_bucket must be a non-empty string",
    )


# ---- named profile resolution ----

PROFILES = json.dumps({
    "pd-qa": {"bucket": "pd-qa-media"},
    "pd-prod": {"bucket": "pd-prod-media", "input_bucket": "pd-prod-uploads"},
    "osg-dev": {"bucket": "osg-media-dev", "access_key_id": "AK-osg", "secret_access_key": "SK-osg"},
    "elsewhere": {"bucket": "far-media", "endpoint_url": "https://other.r2.example",
                  "access_key_id": "AK-far", "secret_access_key": "SK-far"},
})


def test_profile_inherits_endpoint_credentials():
    with _Env(**DEFAULT_ENV, BUCKET_PROFILES=PROFILES):
        t = r2.resolve_bucket_target("pd-prod")
        assert t.profile == "pd-prod"
        assert t.bucket == "pd-prod-media"
        assert t.input_bucket == "pd-prod-uploads"
        assert t.endpoint_url == "https://acct.r2.example"
        assert t.access_key_id == "AK-default"
        assert t.secret_access_key == "SK-default"


def test_profile_input_bucket_defaults_to_bucket():
    with _Env(**DEFAULT_ENV, BUCKET_PROFILES=PROFILES):
        assert r2.resolve_bucket_target("pd-qa").input_bucket == "pd-qa-media"


def test_profile_own_credentials_override_env():
    with _Env(**DEFAULT_ENV, BUCKET_PROFILES=PROFILES):
        t = r2.resolve_bucket_target("osg-dev")
        assert (t.access_key_id, t.secret_access_key) == ("AK-osg", "SK-osg")
        assert t.endpoint_url == "https://acct.r2.example"  # inherited
        t2 = r2.resolve_bucket_target("elsewhere")
        assert t2.endpoint_url == "https://other.r2.example"
        assert t2.access_key_id == "AK-far"


def test_profile_works_on_endpoint_without_default_bucket():
    # An endpoint may carry credentials + profiles only (no default R2_BUCKET_NAME):
    # unnamed jobs are base64, named jobs upload.
    env = dict(DEFAULT_ENV)
    env.pop("R2_BUCKET_NAME")
    with _Env(**env, BUCKET_PROFILES=PROFILES):
        assert r2.resolve_bucket_target("pd-qa").bucket == "pd-qa-media"


def test_unknown_profile_is_rejected_and_lists_allowed():
    with _Env(**DEFAULT_ENV, BUCKET_PROFILES=PROFILES):
        _expect_value_error(lambda: r2.resolve_bucket_target("pd-nope"), "Unknown bucket_profile 'pd-nope'")
        _expect_value_error(lambda: r2.resolve_bucket_target("pd-nope"), "elsewhere, osg-dev, pd-prod, pd-qa")


def test_profile_without_any_profiles_configured_is_rejected():
    with _Env(**DEFAULT_ENV):
        _expect_value_error(lambda: r2.resolve_bucket_target("pd-qa"), "This endpoint allows: none")


def test_profile_without_credentials_anywhere_is_rejected():
    with _Env(BUCKET_PROFILES=PROFILES):
        _expect_value_error(lambda: r2.resolve_bucket_target("pd-qa"), "no complete R2 credentials")
        # ...but a self-contained profile still resolves.
        assert r2.resolve_bucket_target("elsewhere").bucket == "far-media"


def test_profile_never_falls_back_to_default_bucket():
    # A named-but-unknown profile must fail, not silently use R2_BUCKET_NAME.
    with _Env(**DEFAULT_ENV, BUCKET_PROFILES=json.dumps({"pd-qa": {"bucket": "pd-qa-media"}})):
        _expect_value_error(lambda: r2.resolve_bucket_target("pd-prod"), "Unknown bucket_profile")


def test_describe_never_leaks_credentials():
    with _Env(**DEFAULT_ENV, BUCKET_PROFILES=PROFILES):
        for name in (None, "osg-dev"):
            text = r2.resolve_bucket_target(name).describe()
            for secret in ("AK-default", "SK-default", "AK-osg", "SK-osg"):
                assert secret not in text, text


def test_make_uploader_none_for_base64_mode():
    assert r2.make_uploader(None) is None


if __name__ == "__main__":
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

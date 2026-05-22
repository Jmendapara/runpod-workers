"""HTTP + WebSocket client for the in-container ComfyUI server.

Ports the polling, reconnect, and crash-diagnostic logic that was duplicated
across every worker's handler.py.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.parse

import requests
import websocket


COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_API_AVAILABLE_INTERVAL_MS = 50
COMFY_API_AVAILABLE_MAX_RETRIES = 500
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))


def _server_status() -> dict:
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _collect_crash_diagnostics() -> dict:
    diag: dict = {}
    try:
        result = subprocess.run(
            ["pgrep", "-f", "comfyui/main.py"], capture_output=True, text=True, timeout=5
        )
        diag["comfyui_process_alive"] = result.returncode == 0
        if result.stdout.strip():
            diag["comfyui_pids"] = result.stdout.strip().split("\n")
    except Exception as exc:
        diag["comfyui_process_check_error"] = str(exc)

    try:
        result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=5)
        oom_lines = [
            line for line in result.stdout.splitlines()
            if "oom" in line.lower() or "killed process" in line.lower()
                or "out of memory" in line.lower()
        ]
        diag["oom_kill_detected"] = bool(oom_lines)
        if oom_lines:
            diag["oom_messages"] = oom_lines[-5:]
    except Exception as exc:
        diag["dmesg_error"] = str(exc)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,gpu_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            diag["gpu_info"] = result.stdout.strip()
    except Exception as exc:
        diag["nvidia_smi_error"] = str(exc)

    comfy_log = "/var/log/comfyui.log"
    if os.path.exists(comfy_log):
        try:
            result = subprocess.run(
                ["tail", "-n", "50", comfy_log], capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                diag["comfyui_log_tail"] = result.stdout.strip()
        except Exception as exc:
            diag["comfyui_log_error"] = str(exc)

    return diag


def _attempt_reconnect(ws_url: str, initial_error: Exception) -> websocket.WebSocket:
    print(
        f"worker-comfyui - Websocket closed: {initial_error}. Reconnecting...",
        flush=True,
    )
    last_err: Exception = initial_error
    for attempt in range(WEBSOCKET_RECONNECT_ATTEMPTS):
        srv = _server_status()
        if not srv["reachable"]:
            diag = _collect_crash_diagnostics()
            for key, val in diag.items():
                print(f"worker-comfyui - CRASH DIAG [{key}]: {val}", flush=True)
            reason = "ComfyUI process crashed during execution"
            if diag.get("oom_kill_detected"):
                reason = (
                    "ComfyUI was OOM-killed (out of memory). "
                    "Try a GPU with more VRAM or use a smaller/more quantized model."
                )
            elif diag.get("comfyui_process_alive") is False:
                reason = (
                    "ComfyUI process is no longer running (likely crashed). "
                    "Check logs above for CUDA errors or segfaults."
                )
            raise websocket.WebSocketConnectionClosedException(reason)

        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print("worker-comfyui - Websocket reconnected.", flush=True)
            return new_ws
        except (
            websocket.WebSocketException, ConnectionRefusedError, socket.timeout, OSError
        ) as exc:
            last_err = exc
            print(
                f"worker-comfyui - Reconnect attempt {attempt + 1}/{WEBSOCKET_RECONNECT_ATTEMPTS} failed: {exc}",
                flush=True,
            )
            if attempt < WEBSOCKET_RECONNECT_ATTEMPTS - 1:
                time.sleep(WEBSOCKET_RECONNECT_DELAY_S)

    raise websocket.WebSocketConnectionClosedException(
        f"Failed to reconnect after {WEBSOCKET_RECONNECT_ATTEMPTS} attempts. Last error: {last_err}"
    )


class ComfyClient:
    def __init__(self, host: str | None = None):
        self.host = host or COMFY_HOST

    def check_server(
        self,
        retries: int = COMFY_API_AVAILABLE_MAX_RETRIES,
        delay_ms: int = COMFY_API_AVAILABLE_INTERVAL_MS,
    ) -> bool:
        url = f"http://{self.host}/"
        print(f"worker-comfyui - Checking API server at {url}...", flush=True)
        for _ in range(retries):
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    print("worker-comfyui - API is reachable", flush=True)
                    return True
            except (requests.Timeout, requests.RequestException):
                pass
            time.sleep(delay_ms / 1000)
        print(f"worker-comfyui - Failed to connect to {url}", flush=True)
        return False

    def queue_prompt(self, workflow: dict, client_id: str, comfy_org_api_key: str | None = None) -> str:
        payload = {"prompt": workflow, "client_id": client_id}
        env_key = os.environ.get("COMFY_ORG_API_KEY")
        effective_key = comfy_org_api_key or env_key
        if effective_key:
            payload["extra_data"] = {"api_key_comfy_org": effective_key}

        resp = requests.post(
            f"http://{self.host}/prompt",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if resp.status_code == 400:
            try:
                error_data = resp.json()
                message = "Workflow validation failed"
                info = error_data.get("error")
                if isinstance(info, dict):
                    message = info.get("message", message)
                elif info:
                    message = str(info)

                details = []
                for node_id, node_err in (error_data.get("node_errors") or {}).items():
                    if isinstance(node_err, dict):
                        for err_type, err_msg in node_err.items():
                            details.append(f"Node {node_id} ({err_type}): {err_msg}")
                    else:
                        details.append(f"Node {node_id}: {node_err}")

                if details:
                    raise ValueError(message + ":\n" + "\n".join(f"• {d}" for d in details))
                raise ValueError(f"{message}. Raw response: {resp.text}")
            except (json.JSONDecodeError, KeyError):
                raise ValueError(f"ComfyUI validation failed (unparseable): {resp.text}")

        resp.raise_for_status()
        body = resp.json()
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise ValueError(f"Missing 'prompt_id' in queue response: {body}")
        return prompt_id

    def wait_for_completion(self, ws_url: str, prompt_id: str) -> list[str]:
        """Block until the prompt completes or errors. Returns list of error strings."""
        errors: list[str] = []
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...", flush=True)

        try:
            while True:
                try:
                    out = ws.recv()
                except websocket.WebSocketTimeoutException:
                    print("worker-comfyui - WS recv timed out, still waiting...", flush=True)
                    continue
                except websocket.WebSocketConnectionClosedException as closed:
                    ws = _attempt_reconnect(ws_url, closed)
                    continue
                except json.JSONDecodeError:
                    continue

                if not isinstance(out, str):
                    continue

                try:
                    message = json.loads(out)
                except json.JSONDecodeError:
                    continue

                mtype = message.get("type")
                data = message.get("data", {}) or {}

                if mtype == "status":
                    remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining", "N/A")
                    print(f"worker-comfyui - Queue remaining: {remaining}", flush=True)
                elif mtype == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        print(f"worker-comfyui - Execution finished for {prompt_id}", flush=True)
                        return errors
                elif mtype == "execution_error":
                    if data.get("prompt_id") == prompt_id:
                        err = (
                            f"Node Type: {data.get('node_type')}, "
                            f"Node ID: {data.get('node_id')}, "
                            f"Message: {data.get('exception_message')}"
                        )
                        print(f"worker-comfyui - Execution error: {err}", flush=True)
                        errors.append(f"Workflow execution error: {err}")
                        return errors
        finally:
            if ws and ws.connected:
                ws.close()

    def get_history(self, prompt_id: str) -> dict:
        resp = requests.get(f"http://{self.host}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_view(self, filename: str, subfolder: str, file_type: str) -> bytes | None:
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        url = f"http://{self.host}/view?{urllib.parse.urlencode(params)}"
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.content
        except requests.Timeout:
            print(f"worker-comfyui - Timeout fetching {filename}", flush=True)
        except requests.RequestException as exc:
            print(f"worker-comfyui - Error fetching {filename}: {exc}", flush=True)
        except Exception as exc:
            print(f"worker-comfyui - Unexpected error fetching {filename}: {exc}", flush=True)
        return None

    def ws_url(self, client_id: str) -> str:
        return f"ws://{self.host}/ws?clientId={client_id}"

#!/usr/bin/env python3
"""Benchmark a ComfyUI workflow across RunPod serverless GPU pools: cost per JOB, not per hour.

Why: RunPod bills the GPU actually used and treats an endpoint's `gpuTypeIds` as an ordered
preference. A cheaper-per-hour card is often NOT cheaper per job (A6000/A40 run Qwen 2.1 at 2x
the time of a 4090 and end up the dearest pool). Rule for every new model/endpoint: list every
pool the model fits in, cheapest-per-job first, keep the dearer pools as capacity fallbacks.
Measure before choosing. Write-up + past results: opensourcegen/docs/gpu-infra-cost-plan.md.

Two ways to run:

  # A) temporary single-pool endpoints cloned from an existing endpoint's template; benchmarked; deleted
  RUNPOD_API_KEY=... python tests/gpu_pool_bench.py --from-endpoint <endpoint-id> \
      --payload tests/smoke/qwen-image-2-1.json --pools 4090,5090,ada48,amp48,pro6000 \
      --n 6 --override EmptyLatentImage.width=1024 --override KSampler.steps=25

  # B) endpoints that already exist (e.g. prod vs a one-pool test endpoint)
  RUNPOD_API_KEY=... python tests/gpu_pool_bench.py --endpoints h100=<id>,pro6000=<id> \
      --payload tests/smoke/minimax-h3.json --n 5

Job #0 on each pool is the cold start (time-to-first-worker + model load) and is excluded from
the warm median. Every job gets a unique seed (any numeric input key containing "seed") because
identical payloads hit ComfyUI's node cache and "complete" in ~2 s. One job in flight per pool.
Also run the heaviest shape prod allows (max resolution, edit/reference-image workflow) on the
smallest-VRAM pool you intend to list first; an OOM there means that pool goes lower in the list.

Baked-in gotchas: REST-created endpoints MUST carry minCudaVersion or they never schedule;
account quota (maxServerlessConcurrency) is checked before creating anything; temp endpoints are
deleted on exit (Ctrl-C included) unless --keep. New slots can take 2-40 min to appear on busy
pools, hence the long default warm-up timeout.
"""
import argparse, copy, json, os, sys, threading, time, urllib.error, urllib.request

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
RUN = "https://api.runpod.ai/v2"

# $/h = RunPod serverless flex list price (runpod.io/pricing, 2026-11-04). Prices move: re-check and
# pass --rate <pool>=<x>. gpuTypeIds are the REST enum strings; verify with --list-gpus.
# Not serverless pools (create fails "gpuId(s) is required"): RTX PRO 4500 / PRO 5000 Blackwell.
POOLS = {
    "4090":    {"vram": 24,  "rate": 1.34,  "gpus": ["NVIDIA GeForce RTX 4090"]},
    "5090":    {"vram": 32,  "rate": 1.79,  "gpus": ["NVIDIA GeForce RTX 5090"]},
    "amp48":   {"vram": 48,  "rate": 1.22,  "gpus": ["NVIDIA A40", "NVIDIA RTX A6000"]},
    "ada48":   {"vram": 48,  "rate": 1.75,  "gpus": ["NVIDIA RTX 6000 Ada Generation", "NVIDIA L40", "NVIDIA L40S"]},
    "a100":    {"vram": 80,  "rate": 2.72,  "gpus": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]},
    "h100":    {"vram": 80,  "rate": 5.39,  "gpus": ["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]},
    "pro6000": {"vram": 96,  "rate": 3.49,  "gpus": ["NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                                     "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                                                     "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"]},
    "h200":    {"vram": 141, "rate": 6.99,  "gpus": ["NVIDIA H200"]},
    "b200":    {"vram": 180, "rate": 10.79, "gpus": ["NVIDIA B200"]},
}

KEY = ""


def api(method, url, body=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "User-Agent": "runpod-workers-gpu-pool-bench/1.0"})  # rest.runpod.io 403s the default urllib UA
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw.strip() else {}


def gql(query):
    return api("POST", GQL, {"query": query})["data"]


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def load_payload(path, overrides):
    p = json.load(open(path))
    if isinstance(p.get("input"), dict) and "workflow" in p["input"]:
        p = p["input"]
    for k in [k for k in p if k.startswith("_")]:
        p.pop(k)
    if "workflow" not in p:
        die(f"{path}: expected a {{'workflow': ...}} payload (RunPod `input` object)")
    for spec in overrides:
        try:
            lhs, val = spec.split("=", 1)
            cls, k = lhs.split(".", 1)
        except ValueError:
            die(f"bad --override {spec!r}; want Class.key=value")
        try:
            val = json.loads(val)
        except ValueError:
            pass
        hit = 0
        for node in p["workflow"].values():
            if node.get("class_type") == cls:
                node.setdefault("inputs", {})[k] = val
                hit += 1
        print(f"override {cls}.{k}={val!r} -> {hit} node(s)")
        if not hit:
            die(f"--override {spec}: no node with class_type {cls}")
    return p


def run_one(ep, payload, idx, seed, warm_t, job_t):
    p = copy.deepcopy(payload)
    for node in p["workflow"].values():
        for k, v in list(node.get("inputs", {}).items()):
            if "seed" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                node["inputs"][k] = seed
    t0 = time.time()
    try:
        j = api("POST", f"{RUN}/{ep}/run", {"input": p})
    except urllib.error.HTTPError as e:
        return {"i": idx, "seed": seed, "status": f"SUBMIT_HTTP_{e.code}", "err": e.read().decode()[:300]}
    jid = j.get("id")
    limit = warm_t if idx == 0 else job_t
    while True:
        time.sleep(3)
        try:
            s = api("GET", f"{RUN}/{ep}/status/{jid}")
        except Exception as e:  # transient poll errors
            s = {"status": "POLL_ERR", "err": str(e)[:120]}
        st = s.get("status")
        if st in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            o = s.get("output") or {}
            nout = len(o.get("images") or o.get("videos") or []) if isinstance(o, dict) else 0
            err = "" if st == "COMPLETED" else (json.dumps(s.get("error")) if s.get("error") else json.dumps(o))[:300]
            return {"i": idx, "seed": seed, "id": jid, "status": st, "delayTime": s.get("delayTime"),
                    "executionTime": s.get("executionTime"), "wall_s": round(time.time() - t0, 1),
                    "worker": s.get("workerId"), "outputs": nout, "err": err}
        if time.time() - t0 > limit:
            try:
                api("POST", f"{RUN}/{ep}/cancel/{jid}")
            except Exception:
                pass
            return {"i": idx, "seed": seed, "id": jid, "status": "HARNESS_TIMEOUT", "wall_s": round(time.time() - t0, 1)}


def bench(targets, payload, n, warm_t, job_t, out_path):
    results = {name: [] for name in targets}
    lock = threading.Lock()

    def drive(name, pi):
        ep = targets[name]["endpoint"]
        for i in range(n):
            r = run_one(ep, payload, i, 1000 * (pi + 1) + i, warm_t, job_t)
            with lock:
                results[name].append(r)
                json.dump({"targets": targets, "results": results}, open(out_path, "w"), indent=1)
                print(f"[{name}] #{i} {r.get('status')} delay={r.get('delayTime')} exec={r.get('executionTime')} "
                      f"wall={r.get('wall_s')} worker={r.get('worker')} out={r.get('outputs')} {r.get('err', '')}", flush=True)
            if i == 0 and r.get("status") != "COMPLETED":
                print(f"[{name}] cold job failed; skipping this pool", flush=True)
                return

    threads = [threading.Thread(target=drive, args=(nm, i), daemon=True) for i, nm in enumerate(targets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def summarize(results, targets):
    print("\n=== SUMMARY (warm = jobs #1.., cost = median exec x $/h) ===")
    rows = []
    for name, rs in results.items():
        ok = [r for r in rs if r.get("status") == "COMPLETED" and r.get("executionTime")]
        warm = sorted(r["executionTime"] / 1000 for r in ok if r["i"] >= 1)
        cold = next((r for r in ok if r["i"] == 0), None)
        fails = [r for r in rs if r.get("status") != "COMPLETED"]
        if not warm:
            print(f"{name:10} NO warm completions ({len(fails)} failed/timed out): {rs[:1]}")
            continue
        med = warm[len(warm) // 2]
        rate = targets[name]["rate"]
        rows.append({"pool": name, "n": len(warm), "median_s": med, "min_s": warm[0], "max_s": warm[-1],
                     "cold_exec_s": (cold or {}).get("executionTime", 0) / 1000 if cold else None,
                     "cold_delay_s": (cold or {}).get("delayTime", 0) / 1000 if cold else None,
                     "rate": rate, "usd_per_job": med * rate / 3600, "fails": len(fails)})
    if not rows:
        return []
    rows.sort(key=lambda r: r["usd_per_job"])
    best, fastest = rows[0]["usd_per_job"], min(r["median_s"] for r in rows)
    print(f"{'pool':10} {'n':>2} {'median':>7} {'min':>6} {'max':>6} {'cold exec':>9} {'1st worker':>10} {'$/h':>5} {'$/job':>8} {'vs best':>7} {'vs fastest':>10} fails")
    for r in rows:
        print(f"{r['pool']:10} {r['n']:>2} {r['median_s']:7.1f} {r['min_s']:6.1f} {r['max_s']:6.1f} "
              f"{(r['cold_exec_s'] or 0):9.1f} {(r['cold_delay_s'] or 0):10.0f} {r['rate']:5.2f} {r['usd_per_job']:8.4f} "
              f"{r['usd_per_job'] / best:6.2f}x {r['median_s'] / fastest:9.2f}x {r['fails']}")
    order = [r["pool"] for r in rows]
    ids = [g for p in order if p in POOLS for g in POOLS[p]["gpus"]]
    print(f"\nCheapest-per-job order: {order}")
    if ids:
        print("gpuTypeIds for the endpoint (cheapest first; keep all that passed as fallbacks):")
        print(json.dumps(ids))
    return rows


def ensure_quota(need):
    me = gql("{ myself { maxServerlessConcurrency endpoints { workersMax } } }")["myself"]
    used = sum(e["workersMax"] for e in me["endpoints"])
    free = me["maxServerlessConcurrency"] - used
    print(f"serverless quota: {used}/{me['maxServerlessConcurrency']} workersMax used, {free} free, need {need}")
    if free < need:
        die("not enough free worker quota; lower --pools or temporarily reduce some endpoint's workersMax")


def create_endpoints(src_id, pools, prefix, dry_run):
    src = api("GET", f"{REST}/endpoints/{src_id}")
    tpl = src.get("templateId") or (src.get("template") or {}).get("id")
    if not tpl:
        die(f"could not read templateId from endpoint {src_id}")
    base = {
        "computeType": "GPU", "gpuCount": 1, "templateId": tpl,
        "workersMin": 0, "workersMax": 1, "idleTimeout": 60, "flashboot": True,
        "scalerType": "QUEUE_DELAY", "scalerValue": 4,
        "executionTimeoutMs": src.get("executionTimeoutMs") or 900000,
        "minCudaVersion": src.get("minCudaVersion") or "12.0",   # REQUIRED: without it nothing ever schedules
    }
    if src.get("allowedCudaVersions"):
        base["allowedCudaVersions"] = src["allowedCudaVersions"]
    if src.get("networkVolumeId"):
        base["networkVolumeId"] = src["networkVolumeId"]
        print("warning: source uses a network volume; pools outside its data center will never get workers")
    print(f"cloning template {tpl} from {src.get('name')} (cuda>={base['minCudaVersion']}, timeout {base['executionTimeoutMs']} ms)")
    created, targets = [], {}
    stamp = time.strftime("%m%d%H%M")
    for pool in pools:
        body = dict(base, name=f"{prefix}-{pool}-{stamp}", gpuTypeIds=POOLS[pool]["gpus"])
        if dry_run:
            print("DRY RUN create:", json.dumps(body))
            continue
        r = api("POST", f"{REST}/endpoints", body)
        if not r.get("id"):
            print(f"create failed for {pool}: {json.dumps(r)[:300]}")
            continue
        created.append(r["id"])
        targets[pool] = {"endpoint": r["id"], "rate": POOLS[pool]["rate"]}
        print(f"created {body['name']} = {r['id']}")
    return created, targets


def delete_endpoints(ids):
    for eid in ids:
        for attempt in range(4):
            try:
                api("PATCH", f"{REST}/endpoints/{eid}", {"workersMin": 0, "workersMax": 0})
                api("DELETE", f"{REST}/endpoints/{eid}")
                print(f"deleted {eid}")
                break
            except Exception as e:
                print(f"delete {eid} attempt {attempt + 1} failed: {str(e)[:120]}")
                time.sleep(10)
        else:
            print(f"!! could not delete {eid}; remove it in the RunPod console")


def main():
    global KEY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", help="RunPod input JSON ({'workflow': ...}), e.g. tests/smoke/<model>.json")
    ap.add_argument("--from-endpoint", help="existing endpoint whose template/CUDA settings the temp endpoints clone")
    ap.add_argument("--pools", help="comma list of pool names to create (see --list-pools)")
    ap.add_argument("--endpoints", help="bench existing endpoints instead: name=id,name=id (name in POOLS gives the rate)")
    ap.add_argument("--rate", action="append", default=[], help="override $/h: pool=3.49 (repeatable)")
    ap.add_argument("--override", action="append", default=[], help="Class.key=value applied to every node of that class_type")
    ap.add_argument("--n", type=int, default=6, help="jobs per pool incl. the cold one (default 6)")
    ap.add_argument("--warmup-timeout", type=int, default=2400, help="seconds to wait for the cold job (default 2400)")
    ap.add_argument("--job-timeout", type=int, default=900, help="seconds per warm job (default 900)")
    ap.add_argument("--out", help="results JSON path (default bench_results/<payload>_<ts>.json)")
    ap.add_argument("--name-prefix", default="bench", help="temp endpoint name prefix")
    ap.add_argument("--keep", action="store_true", help="do not delete the temp endpoints")
    ap.add_argument("--dry-run", action="store_true", help="print the create bodies and exit")
    ap.add_argument("--list-pools", action="store_true")
    ap.add_argument("--list-gpus", action="store_true", help="print RunPod's gpuTypes (id, VRAM) and exit")
    a = ap.parse_args()

    if a.list_pools:
        for k, v in POOLS.items():
            print(f"{k:8} {v['vram']:>3} GB  ${v['rate']:5.2f}/h  {v['gpus']}")
        return
    KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not KEY:
        die("set RUNPOD_API_KEY")
    if a.list_gpus:
        for g in gql("{ gpuTypes { id displayName memoryInGb secureCloud communityCloud } }")["gpuTypes"]:
            print(f"{g['memoryInGb']:>4} GB  {g['id']}")
        return
    if not a.payload:
        die("--payload is required")
    for spec in a.rate:
        pool, val = spec.split("=", 1)
        POOLS.setdefault(pool, {"vram": 0, "gpus": []})["rate"] = float(val)
    payload = load_payload(a.payload, a.override)
    out = a.out or os.path.join("bench_results", f"{os.path.splitext(os.path.basename(a.payload))[0]}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    created, targets = [], {}
    if a.endpoints:
        for item in a.endpoints.split(","):
            name, eid = item.split("=", 1)
            rate = POOLS.get(name, {}).get("rate")
            if rate is None:
                die(f"no $/h known for {name!r}; pass --rate {name}=<x>")
            targets[name] = {"endpoint": eid.strip(), "rate": rate}
    elif a.from_endpoint:
        if not a.pools:
            die("--pools is required with --from-endpoint")
        pools = [p.strip() for p in a.pools.split(",") if p.strip()]
        bad = [p for p in pools if p not in POOLS or not POOLS[p]["gpus"]]
        if bad:
            die(f"unknown pool(s) {bad}; see --list-pools")
        if not a.dry_run:
            ensure_quota(len(pools))
        created, targets = create_endpoints(a.from_endpoint, pools, a.name_prefix, a.dry_run)
        if a.dry_run:
            return
    else:
        die("pass --from-endpoint + --pools, or --endpoints")
    if not targets:
        die("nothing to benchmark")

    print(f"benchmarking {len(targets)} pool(s), {a.n} jobs each, results -> {out}")
    try:
        results = bench(targets, payload, a.n, a.warmup_timeout, a.job_timeout, out)
        rows = summarize(results, targets)
        json.dump({"targets": targets, "results": results, "summary": rows}, open(out, "w"), indent=1)
    finally:
        if created and not a.keep:
            print("cleaning up temp endpoints...")
            delete_endpoints(created)
        elif created:
            print(f"kept temp endpoints: {created}")


if __name__ == "__main__":
    main()

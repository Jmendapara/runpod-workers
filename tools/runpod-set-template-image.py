#!/usr/bin/env python3
"""Swap the container image on a RunPod serverless endpoint's template.

Endpoint templates named `…__template__…` are invisible to the REST /v1/templates
API, so this uses the GraphQL `saveTemplate` mutation (full field set + env must be
resent, and a User-Agent header is required or the API answers 403).

Usage:
  RUNPOD_API_KEY=… tools/runpod-set-template-image.py <endpoint-id> <image> --expect-name "PD - Z Image Turbo - Dev" [-y]

Without -y it only prints the endpoint's template name + current image (dry run).
`--expect-name` must be a substring of the template name — a guard against pasting
the wrong endpoint id. After the swap, RECYCLE the workers
(tools/runpod-recycle-workers.sh) or idle/FlashBoot workers keep the old image.
"""
import argparse
import json
import os
import sys
import urllib.request

GQL = "https://api.runpod.io/graphql"
TEMPLATE_FIELDS = [
    "id", "name", "imageName", "containerDiskInGb", "volumeInGb", "volumeMountPath",
    "dockerArgs", "ports", "readme", "isServerless", "startSsh", "startJupyter",
    "containerRegistryAuthId",
]


def gql(key, query, variables=None):
    req = urllib.request.Request(
        GQL,
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "runpod-workers-tools/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp)
    if body.get("errors"):
        raise SystemExit(f"GraphQL errors: {json.dumps(body['errors'])}")
    return body["data"]


def endpoint_template(key, endpoint_id):
    fields = " ".join(TEMPLATE_FIELDS)
    data = gql(key, "{ myself { endpoints { id template { %s env { key value } } } } }" % fields)
    for ep in data["myself"]["endpoints"]:
        if ep["id"] == endpoint_id:
            return ep["template"]
    raise SystemExit(f"endpoint {endpoint_id} not found in this account")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("endpoint_id")
    ap.add_argument("image", help="full image ref, e.g. jmendapara/z-image-turbo-runpod-worker:2026-09-22-1234-abcdef0")
    ap.add_argument("--expect-name", required=True, help="substring the template name must contain")
    ap.add_argument("-y", "--yes", action="store_true", help="apply the change (default: dry run)")
    args = ap.parse_args()

    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("set RUNPOD_API_KEY")

    tpl = endpoint_template(key, args.endpoint_id)
    print(f"endpoint  : {args.endpoint_id}")
    print(f"template  : {tpl['id']}  {tpl['name']}")
    print(f"current   : {tpl['imageName']}")
    print(f"requested : {args.image}")
    if args.expect_name not in (tpl["name"] or ""):
        sys.exit(f"REFUSING: template name does not contain {args.expect_name!r}")
    if tpl["imageName"] == args.image:
        print("already on the requested image — nothing to do")
        return
    if not args.yes:
        print("dry run — re-run with -y to apply")
        return

    inp = {k: tpl[k] for k in TEMPLATE_FIELDS}
    inp["imageName"] = args.image
    inp["readme"] = inp["readme"] or ""
    inp["env"] = tpl["env"]
    gql(key, "mutation($input: SaveTemplateInput!) { saveTemplate(input: $input) { id imageName } }", {"input": inp})

    after = endpoint_template(key, args.endpoint_id)
    print(f"now       : {after['imageName']}")
    if after["imageName"] != args.image:
        sys.exit("swap did not stick — re-read shows a different image")
    print("done — now recycle the workers: tools/runpod-recycle-workers.sh", args.endpoint_id)


if __name__ == "__main__":
    main()

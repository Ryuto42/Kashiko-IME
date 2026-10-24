#!/usr/bin/env python3
"""Print the licenses of every crate linked into kashiko-imserver.

Walks the normal (non-dev, non-build) dependency graph of kashiko-im from
`cargo metadata` and prints each crate's license expression followed by the
license files it ships. Used by package-release.sh to put the full texts in
the release zip.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "unlicense")
# License files that live below a -sys crate's root, for the C library it builds.
NESTED = {"onig_sys": ["oniguruma/COPYING"]}


def main():
    meta = json.loads(
        subprocess.check_output(
            [
                "cargo", "metadata", "--format-version", "1",
                "--filter-platform", "aarch64-apple-darwin",
                "--manifest-path", os.path.join(ROOT, "Cargo.toml"),
            ]
        )
    )
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    workspace = set(meta["workspace_members"])
    start = next(i for i in workspace if packages[i]["name"] == "kashiko-im")

    seen, stack = set(), [start]
    while stack:
        node = nodes[stack.pop()]
        for dep in node["deps"]:
            normal = any(k["kind"] is None for k in dep["dep_kinds"])
            if normal and dep["pkg"] not in seen:
                seen.add(dep["pkg"])
                stack.append(dep["pkg"])

    out = sys.stdout
    out.write("Licenses of the Rust crates linked into kashiko-imserver\n")
    out.write("=" * 79 + "\n\n")
    for pkg in sorted((packages[i] for i in seen if i not in workspace), key=lambda p: (p["name"], p["version"])):
        root = os.path.dirname(pkg["manifest_path"])
        out.write("-" * 79 + "\n")
        out.write(f"{pkg['name']} {pkg['version']}\n")
        out.write(f"License: {pkg.get('license') or pkg.get('license_file') or 'unknown'}\n")
        if pkg.get("repository"):
            out.write(f"Repository: {pkg['repository']}\n")
        out.write("-" * 79 + "\n\n")
        files = sorted(f for f in os.listdir(root) if f.lower().startswith(LICENSE_PREFIXES))
        files += [f for f in NESTED.get(pkg["name"], []) if os.path.exists(os.path.join(root, f))]
        for name in files:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    out.write(f"--- {name} ---\n{f.read().rstrip()}\n\n")


if __name__ == "__main__":
    main()

"""E3/T-110b regression guard: no relative editable sibling source, direct ref.

scholar-graph-kit declared ``"scholar-search-kit"`` as a BARE name in its base
``project.dependencies`` that was only resolvable through the relative editable
``[tool.uv.sources]`` entry::

    scholar-search-kit = { path = "../scholar-search-kit", editable = true }

That entry resolves only inside the harness monorepo (or beside a sibling
checkout on disk). For a standalone ``uv pip install`` of this kit's wheel the
requirement was unresolvable -- the wheel METADATA carried a bare
``Requires-Dist: scholar-search-kit`` and no index serves these kits -- and for a
git checkout of this kit alone the relative path does not exist at all, so
pip/uv aborts with ``has no subdirectory '../scholar-search-kit'``.

This test locks in the T-110b fix: the sibling is a PEP 508 direct git reference
pinned to a full 40-hex canonical SHA and no relative path source may come back.
It is hermetic -- it reads the checked-in ``pyproject.toml`` only, never touches
the network, and never invokes ``uv``. When a wheel has already been built into
``dist/`` the built METADATA is additionally checked.

The scan covers ``project.dependencies`` **and** every
``project.optional-dependencies`` group, so relocating the sibling into an extra
cannot make this test pass vacuously.
"""

from __future__ import annotations

import re
import tomllib
import zipfile
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# scholar-<name>[@ git+https://github.com/nexus-scholar-org/scholar-<name>@<40-hex>]
# with an optional PEP 508 extras suffix, e.g. scholar-pdf-kit[extract].
DIRECT_REF = re.compile(
    r"^scholar-[a-z0-9-]+(\[[a-z0-9,.-]+\])? @ git\+https://github\.com/nexus-scholar-org/scholar-[a-z0-9-]+@[0-9a-f]{40}$"
)

# The canonical main SHA recorded at E3/T-110 dispatch time.
EXPECTED_SHA = {
    "scholar-search-kit": "911d864fcb6a706d4c0339f80524a46f591e2cad",
}


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _all_declared_requirements() -> list[str]:
    """Every declared requirement: base deps plus every optional-extra group."""
    project = _load()["project"]
    reqs = list(project.get("dependencies", []))
    for extra_reqs in project.get("optional-dependencies", {}).values():
        reqs.extend(extra_reqs)
    return reqs


def test_every_sibling_is_a_sha_pinned_direct_git_reference() -> None:
    reqs = _all_declared_requirements()
    siblings = [d for d in reqs if d.startswith("scholar-")]
    assert siblings, "expected declared scholar-* sibling requirements"

    for dep in siblings:
        assert DIRECT_REF.match(dep), f"not a SHA-pinned direct git reference: {dep!r}"

    declared = {re.split(r"[ @\[]", d, maxsplit=1)[0]: d for d in siblings}
    assert set(declared) == set(EXPECTED_SHA), (
        f"unexpected sibling set: {sorted(declared)}"
    )
    for name, sha in EXPECTED_SHA.items():
        assert declared[name].endswith(sha), (
            f"{name} is not pinned to canonical main {sha}: {declared[name]!r}"
        )


def test_sibling_is_a_hard_runtime_dependency() -> None:
    """The ref is a base dependency here, not an extra; keep it that way."""
    deps = _load()["project"]["dependencies"]
    matches = [d for d in deps if d.startswith("scholar-search-kit")]
    assert len(matches) == 1, (
        f"expected exactly one base-dependency scholar-search-kit ref: {matches}"
    )
    assert matches[0].endswith(EXPECTED_SHA["scholar-search-kit"]), (
        f"base dependency is not pinned to canonical main: {matches[0]!r}"
    )


def test_no_relative_editable_sibling_source_can_come_back() -> None:
    sources = _load().get("tool", {}).get("uv", {}).get("sources", {})
    relative = {
        name: src
        for name, src in sources.items()
        if isinstance(src, dict) and "path" in src
    }
    assert not relative, (
        f"relative sibling sources break standalone installs: {relative}"
    )


def test_direct_reference_opt_in_is_present() -> None:
    """hatchling refuses a direct reference without this opt-in."""
    assert (
        _load()
        .get("tool", {})
        .get("hatch", {})
        .get("metadata", {})
        .get("allow-direct-references")
        is True
    ), "[tool.hatch.metadata] allow-direct-references must be true"


def test_built_wheel_metadata_carries_the_direct_refs() -> None:
    wheels = sorted((PYPROJECT.parent / "dist").glob("*.whl"))
    if not wheels:
        import pytest

        pytest.skip("no built wheel in dist/; run `uv build --wheel .` first")

    archive = zipfile.ZipFile(wheels[-1])
    metadata_name = next(
        n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
    )
    requires = [
        line.removeprefix("Requires-Dist: ")
        for line in archive.read(metadata_name).decode().splitlines()
        if line.startswith("Requires-Dist: ")
    ]

    # A bare sibling name in METADATA is exactly the bug: unresolvable outside
    # the monorepo because no index serves these kits.
    bare = [r for r in requires if "scholar-" in r and "git+" not in r]
    assert not bare, f"bare scholar-* Requires-Dist is unresolvable: {bare}"

    git_requires = [r for r in requires if "git+" in r]
    assert len(git_requires) == len(EXPECTED_SHA), (
        f"expected {len(EXPECTED_SHA)} git direct refs, got {git_requires}"
    )
    for name, sha in EXPECTED_SHA.items():
        # hatchling preserves the PEP 508 `name @ git+...` spacing (and would
        # append `; extra == "..."` if the ref ever moved into an extra), so the
        # ref is matched as an exact substring rather than via endswith.
        expected_ref = f"{name} @ git+https://github.com/nexus-scholar-org/{name}@{sha}"
        assert any(expected_ref in r for r in git_requires), (
            f"missing {name}@{sha} in METADATA: {git_requires}"
        )

#!/usr/bin/env bash
# Release checklist for feral.
#
# A feral release version lives in SIX places that must move together.
# The root crate and the Python bindings are SEPARATE cargo workspaces
# (python/Cargo.toml has its own empty [workspace]), so a root-only
# `cargo` invocation never touches the Python lockfile. The v0.5.0
# release shipped with the Python trio left at 0.4.0; the wheels built
# as feral_solver-0.4.0 and PyPI rejected the upload (a version's files
# cannot be re-uploaded). This script keeps the six in sync.
#
# Version-bearing locations (canonical = root Cargo.toml [package]):
#   1. Cargo.toml              [package] version       (feral)
#   2. Cargo.lock              feral entry
#   3. python/Cargo.toml       [package] version       (feral-python)
#   4. python/pyproject.toml   [project] version       (feral-solver)
#   5. python/Cargo.lock       feral-python entry
#   6. python/Cargo.lock       feral entry (path dep)
#
# The six fill-reducing ordering crates (feral-amd, …) version
# INDEPENDENTLY of feral and are never *modified* here, but `check` also
# guards against ordering-crate STALENESS: release.yml publishes them in
# the same run and treats "already on crates.io" as success, so a crate
# whose code changed since the last release tag without a version bump is
# SILENTLY SKIPPED at publish time — shipping stale code (the same drift
# that shipped the v0.5.0 Python trio at 0.4.0). See do_ordering_check.
#
# Usage:
#   scripts/release-checklist.sh [check]      verify all six agree (default)
#   scripts/release-checklist.sh bump X.Y.Z   set all six to X.Y.Z, re-check
#
# Exit 0 if consistent, 1 if a mismatch / problem is found. The check
# form is safe to wire into CI as a release-readiness guard.

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

# --- extractors --------------------------------------------------------

# Version of `version = "..."` inside a named [section] of a TOML file.
toml_version() {
    local file=$1 section=$2
    awk -v sec="[$section]" '
        $0 == sec        { ins = 1; next }
        /^\[/            { ins = 0 }
        ins && /^version[[:space:]]*=/ {
            match($0, /"[^"]*"/); print substr($0, RSTART + 1, RLENGTH - 2); exit
        }
    ' "$file"
}

# Version of the first [[package]] named `pkg` in a Cargo.lock file.
lock_version() {
    local file=$1 pkg=$2
    awk -v want="name = \"$pkg\"" '
        $0 == want { found = 1; next }
        found && /^version[[:space:]]*=/ {
            match($0, /"[^"]*"/); print substr($0, RSTART + 1, RLENGTH - 2); exit
        }
    ' "$file"
}

# --- ordering-crate staleness guard ------------------------------------

# [package] version recorded at <ref>:<path>; empty if the path or the
# [package] section is absent at that ref. The trailing `|| true` swallows
# `git show`'s failure (a path that does not exist at <ref>) so the
# pipefail/set -e combination does not abort the script.
git_show_pkg_version() {
    local ref=$1 path=$2
    git show "$ref:$path" 2>/dev/null | awk '
        $0 == "[package]" { ins = 1; next }
        /^\[/            { ins = 0 }
        ins && /^version[[:space:]]*=/ {
            match($0, /"[^"]*"/); print substr($0, RSTART + 1, RLENGTH - 2); exit
        }
    ' || true
}

# Flag any publishable crate under crates/ whose tree changed since the
# last vX.Y.Z tag while its [package] version stayed put — release.yml
# would skip it and publish stale code. Read-only: never mutates crates/.
# Returns 1 if at least one crate is stale, else 0.
do_ordering_check() {
    local last_tag stale=0
    last_tag="$(git describe --tags --abbrev=0 --match 'v*' 2>/dev/null || true)"

    echo "ordering-crate staleness (vs last release tag)"
    if [ -z "$last_tag" ]; then
        echo "  note  no vX.Y.Z tag yet — first release, nothing to compare"
        echo
        return 0
    fi
    echo "  last release tag: $last_tag"

    local manifest dir vtag vhead
    for manifest in crates/*/Cargo.toml; do
        # Skip crates excluded from publication (publish = false).
        if grep -qE '^[[:space:]]*publish[[:space:]]*=[[:space:]]*false' "$manifest"; then
            continue
        fi
        dir="$(basename "$(dirname "$manifest")")"
        vhead="$(toml_version "$manifest" package)"
        vtag="$(git_show_pkg_version "$last_tag" "crates/$dir/Cargo.toml")"

        if [ -z "$vtag" ]; then
            # Did not exist at the last tag — brand new, will publish fresh.
            printf '  ok    %-22s new since %s (v%s)\n' "$dir" "$last_tag" "$vhead"
        elif git diff --quiet "$last_tag"..HEAD -- "crates/$dir"; then
            printf '  ok    %-22s unchanged since %s (v%s)\n' "$dir" "$last_tag" "$vhead"
        elif [ "$vtag" = "$vhead" ]; then
            printf '  STALE %-22s changed since %s but still v%s — bump it\n' \
                "$dir" "$last_tag" "$vhead"
            stale=1
        else
            printf '  ok    %-22s changed, bumped v%s -> v%s\n' "$dir" "$vtag" "$vhead"
        fi
    done
    echo

    [ "$stale" -eq 0 ]
}

# --- the six readings --------------------------------------------------
# Each row: "label|reading". The canonical version is row 1.

readings() {
    printf '%s\n' \
        "Cargo.toml [package]|$(toml_version Cargo.toml package)" \
        "Cargo.lock feral|$(lock_version Cargo.lock feral)" \
        "python/Cargo.toml [package]|$(toml_version python/Cargo.toml package)" \
        "python/pyproject.toml [project]|$(toml_version python/pyproject.toml project)" \
        "python/Cargo.lock feral-python|$(lock_version python/Cargo.lock feral-python)" \
        "python/Cargo.lock feral|$(lock_version python/Cargo.lock feral)"
}

# --- check -------------------------------------------------------------

do_check() {
    local canonical="" versions_ok=1 changelog_ok=1
    canonical="$(toml_version Cargo.toml package)"

    echo "feral release version check"
    echo "  canonical (Cargo.toml [package]): ${canonical:-<missing>}"
    echo
    while IFS='|' read -r label reading; do
        if [ "$reading" = "$canonical" ] && [ -n "$reading" ]; then
            printf '  ok    %-34s %s\n' "$label" "$reading"
        else
            printf '  DIFF  %-34s %s\n' "$label" "${reading:-<missing>}"
            versions_ok=0
        fi
    done < <(readings)
    echo

    # CHANGELOG must have a dated section for this version.
    if grep -qE "^## \[$canonical\]" CHANGELOG.md; then
        echo "  ok    CHANGELOG.md has a [$canonical] section"
    else
        echo "  WARN  CHANGELOG.md has no [$canonical] section yet"
        changelog_ok=0
    fi

    # The tag should NOT exist yet for an unreleased version.
    if git rev-parse -q --verify "refs/tags/v$canonical" >/dev/null; then
        echo "  note  tag v$canonical already exists (already released?)"
    else
        echo "  ok    tag v$canonical does not exist yet"
    fi

    echo

    # Ordering-crate staleness guard (read-only; never mutates crates/).
    local ordering_ok=1
    do_ordering_check || ordering_ok=0

    local rc=0
    if [ "$versions_ok" -eq 0 ]; then
        echo "RESULT: version strings disagree — run"
        echo "        'scripts/release-checklist.sh bump $canonical' (or the"
        echo "        intended version) to sync all six."
        rc=1
    fi
    if [ "$ordering_ok" -eq 0 ]; then
        echo "RESULT: an ordering crate changed since the last release but its"
        echo "        version did not move — release.yml would SILENTLY SKIP it,"
        echo "        publishing stale code. Bump the flagged crate(s) above and"
        echo "        record it in CHANGELOG.md."
        rc=1
    fi
    if [ "$changelog_ok" -eq 0 ]; then
        echo "RESULT: CHANGELOG.md still needs a [$canonical] section."
        rc=1
    fi
    if [ "$rc" -eq 0 ]; then
        echo "RESULT: all six version strings agree on $canonical; no ordering"
        echo "        crate is stale; CHANGELOG has a [$canonical] section."
    fi
    return "$rc"
}

# --- bump --------------------------------------------------------------

# Rewrite `version = "..."` inside [section] of a TOML file, in place.
set_toml_version() {
    local file=$1 section=$2 ver=$3 tmp
    tmp="$(mktemp)"
    awk -v sec="[$section]" -v ver="$ver" '
        $0 == sec { ins = 1; print; next }
        /^\[/     { ins = 0 }
        ins && /^version[[:space:]]*=/ { print "version = \"" ver "\""; next }
        { print }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

do_bump() {
    local ver=$1
    if ! [[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "error: version must be X.Y.Z (got '$ver')" >&2
        exit 1
    fi

    echo "bumping all six version strings to $ver ..."
    set_toml_version Cargo.toml             package "$ver"
    set_toml_version python/Cargo.toml      package "$ver"
    set_toml_version python/pyproject.toml  project "$ver"

    # Refresh both lockfiles. --workspace updates only workspace members
    # (feral / feral-python), never their third-party dependencies.
    echo "  cargo update --workspace (root) ..."
    cargo update --manifest-path Cargo.toml --workspace >/dev/null 2>&1
    echo "  cargo update --workspace (python) ..."
    cargo update --manifest-path python/Cargo.toml --workspace >/dev/null 2>&1
    echo

    do_check || true   # report the new state; CHANGELOG is still manual

    cat <<EOF

Remaining manual steps (this script does NOT commit, tag, or release):

  1. Edit CHANGELOG.md — move the [Unreleased] entries into a new
     section:  ## [$ver] - $(date +%Y-%m-%d)
  2. cargo test            (root workspace — hard rule before commit)
     cargo test --manifest-path python/Cargo.toml
  3. Review:  git diff
  4. git commit  (every commit needs a body: what / why / evidence)
  5. git tag -a v$ver -m "feral v$ver"
  6. git push origin main && git push origin v$ver
  7. gh release create v$ver --title "feral v$ver" --notes-file <notes> \\
       --verify-tag

  ⚠  Step 7 is the irreversible trigger. Publishing the GitHub release
     fires release.yml (publishes all 7 crates to crates.io — permanent,
     only 'cargo yank' is possible) AND python-wheels.yml (builds
     feral_solver-$ver wheels and publishes to PyPI — that version's
     files can never be re-uploaded). Make sure $ver is final first.

  8. Verify (crates.io rejects a request with no User-Agent, and does it
     silently — without the header even 'serde' reads as unpublished):
       curl -s -H 'User-Agent: feral-release (jkitchin@andrew.cmu.edu)' \\
         https://crates.io/api/v1/crates/feral | \\
         python3 -c 'import sys,json;print(json.load(sys.stdin)["crate"]["max_version"])'
       curl -s https://pypi.org/pypi/feral-solver/json | \\
         python3 -c 'import sys,json;print(json.load(sys.stdin)["info"]["version"])'
EOF
}

# --- dispatch ----------------------------------------------------------

case "${1:-check}" in
    check)
        do_check
        ;;
    bump)
        if [ $# -lt 2 ]; then
            echo "usage: scripts/release-checklist.sh bump X.Y.Z" >&2
            exit 1
        fi
        do_bump "$2"
        ;;
    -h|--help|help)
        sed -n '2,30p' "$0"
        ;;
    *)
        echo "usage: scripts/release-checklist.sh [check | bump X.Y.Z]" >&2
        exit 1
        ;;
esac

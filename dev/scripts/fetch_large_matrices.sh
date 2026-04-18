#!/usr/bin/env bash
# Fetch the large-matrix corpus from the SuiteSparse Matrix Collection.
#
# Downloads the matrices listed in dev/scripts/large_matrices.txt into
# tests/data/large/<name>.mtx. The archive (tar.gz) is removed after
# extraction; only the .mtx file is kept.
#
# Re-running is safe: matrices already present on disk are skipped.
#
# Total size: ~45 MB download, ~150 MB unpacked.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${REPO_ROOT}/dev/scripts/large_matrices.txt"
DEST="${REPO_ROOT}/tests/data/large"
MIRROR="https://suitesparse-collection-website.herokuapp.com/MM"

mkdir -p "${DEST}"

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
    exit 1
fi

while IFS= read -r line; do
    # Skip comments and blank lines.
    case "${line}" in
        \#*|"") continue ;;
    esac
    group="${line%%/*}"
    name="${line##*/}"
    target="${DEST}/${name}.mtx"
    if [[ -f "${target}" ]]; then
        echo "skip ${name} (already present)"
        continue
    fi
    url="${MIRROR}/${group}/${name}.tar.gz"
    tmp="${DEST}/${name}.tar.gz"
    echo "fetch ${url}"
    curl -fL --retry 3 -o "${tmp}" "${url}"
    # The archive lays out files under ./<name>/<name>.mtx.
    tar -xzf "${tmp}" -C "${DEST}"
    mv "${DEST}/${name}/${name}.mtx" "${target}"
    rm -rf "${DEST:?}/${name}"
    rm -f "${tmp}"
    echo "done ${name} ($(wc -c <"${target}") bytes)"
done < "${MANIFEST}"

echo "all matrices under ${DEST}"

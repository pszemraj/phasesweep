#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/phasesweep-wheel-smoke.XXXXXX")"
trap 'rm -rf -- "$smoke_root"' EXIT

wheel_dir="$smoke_root/wheel"
install_root="$smoke_root/install"
project_dir="$smoke_root/project"
source_root="$smoke_root/source"
mkdir -p "$wheel_dir" "$install_root" "$project_dir"

git clone --quiet --no-hardlinks "$repo_root" "$source_root"
git -C "$repo_root" ls-files --cached --others --exclude-standard -z \
  | tar -C "$repo_root" --null --files-from=- -cf - \
  | tar -C "$source_root" -xf -
pip wheel --no-deps --wheel-dir "$wheel_dir" "$source_root"
wheel_files=("$wheel_dir"/*.whl)
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Expected exactly one wheel in $wheel_dir" >&2
  exit 1
fi
pip install --ignore-installed --no-deps --prefix "$install_root" "${wheel_files[0]}"

site_packages="$(find "$install_root/lib" -type d -name site-packages -print -quit)"
if [[ -z "$site_packages" ]]; then
  echo "Installed wheel did not create a site-packages directory" >&2
  exit 1
fi

export PATH="$install_root/bin:$PATH"
export PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir"

phasesweep init
phasesweep validate experiment.yaml
phasesweep run experiment.yaml --dry-run
phasesweep mcp init-catalog --from experiment.yaml -o catalog.yaml

test -f experiment.yaml
test -f catalog.yaml
echo "Installed-wheel starter smoke passed."

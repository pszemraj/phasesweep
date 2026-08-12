#!/usr/bin/env bash
# Build the current working tree into a wheel, install it into a throwaway
# prefix, and exercise the installed starter workflow. Every artifact lives
# under a temporary root that is removed on exit; the repository is untouched.
set -euo pipefail

fail() {
  echo "check_installed_wheel: $*" >&2
  exit 1
}

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" \
  || fail "could not resolve the repository root from ${BASH_SOURCE[0]}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/phasesweep-wheel-smoke.XXXXXX")" \
  || fail "could not create a temporary smoke root under ${TMPDIR:-/tmp}"
trap 'rm -rf -- "$smoke_root"' EXIT

wheel_dir="$smoke_root/wheel"
install_root="$smoke_root/install"
project_dir="$smoke_root/project"
source_root="$smoke_root/source"
mkdir -p "$wheel_dir" "$install_root" "$project_dir" \
  || fail "could not create the smoke directories under $smoke_root"

command -v git >/dev/null || fail "git is not on PATH"
command -v pip >/dev/null || fail "pip is not on PATH; activate the project environment first"
command -v python >/dev/null || fail "python is not on PATH; activate the project environment first"

# Ground truth for the version setuptools-scm must derive in the fresh clone.
describe_args=(describe --dirty --tags --long --match '*[0-9]*')
repo_describe="$(git -C "$repo_root" "${describe_args[@]}")" \
  || fail "git ${describe_args[*]} failed in $repo_root; cannot establish an expected version"
head_hash="$(git -C "$repo_root" rev-parse HEAD)" \
  || fail "git rev-parse HEAD failed in $repo_root"

git clone --quiet --no-hardlinks "$repo_root" "$source_root" \
  || fail "git clone of $repo_root into $source_root failed"

# Overlay the working tree (uncommitted edits included) onto the fresh clone so
# the wheel reflects what is on disk. Submodule paths are excluded on purpose:
# tar would otherwise copy the initialized submodule's `.git` gitlink file,
# which points at "$repo_root/.git/modules/..." and does not exist in the clone.
# Every git command that walks that path then exits 128, setuptools-scm loses
# `git describe`, and it silently falls back to a bogus 0.1.dev<commits>
# version. The clone's own git metadata must stay exactly as git wrote it.
tar_excludes=()
while IFS= read -r -d '' entry; do
  case "$entry" in
    "160000 "*) tar_excludes+=(--exclude="${entry#*$'\t'}") ;;
  esac
done < <(git -C "$repo_root" ls-files --stage -z)

# GNU tar applies --exclude only to names it reads afterwards, so the excludes
# must precede --files-from or they are silently ignored.
if ((${#tar_excludes[@]})); then
  git -C "$repo_root" ls-files --cached --others --exclude-standard -z \
    | tar -C "$repo_root" "${tar_excludes[@]}" --null --files-from=- -cf - \
    | tar -C "$source_root" -xf - \
    || fail "working-tree overlay from $repo_root into $source_root failed"
else
  git -C "$repo_root" ls-files --cached --others --exclude-standard -z \
    | tar -C "$repo_root" --null --files-from=- -cf - \
    | tar -C "$source_root" -xf - \
    || fail "working-tree overlay from $repo_root into $source_root failed"
fi

if ((${#tar_excludes[@]})); then
  for exclude in "${tar_excludes[@]}"; do
    gitlink="$source_root/${exclude#--exclude=}/.git"
    [[ ! -e "$gitlink" ]] \
      || fail "submodule gitlink leaked into the fresh clone at $gitlink; setuptools-scm would see broken git metadata"
  done
fi

source_describe="$(git -C "$source_root" "${describe_args[@]}")" \
  || fail "git ${describe_args[*]} failed in the overlaid clone $source_root; the clone's git metadata is broken and setuptools-scm would fall back to a wrong version"
# Compare the tag/distance/node lineage only. The `-dirty` suffix may legitimately
# differ: the clone leaves submodules uninitialized, so a submodule checked out at
# a non-recorded commit dirties the source checkout but not the clone. It does not
# affect which release setuptools-scm derives.
[[ "${source_describe%-dirty}" == "${repo_describe%-dirty}" ]] \
  || fail "overlaid clone describes as '$source_describe' but $repo_root describes as '$repo_describe'"

describe_pattern='^(.+)-([0-9]+)-g([0-9a-fA-F]+)(-dirty)?$'
[[ "$source_describe" =~ $describe_pattern ]] \
  || fail "could not parse 'git describe --long' output '$source_describe'"
scm_tag="${BASH_REMATCH[1]}"
scm_distance="${BASH_REMATCH[2]}"
tag_version="${scm_tag#v}"

pip wheel --no-deps --wheel-dir "$wheel_dir" "$source_root" \
  || fail "pip wheel failed for $source_root"
wheel_files=("$wheel_dir"/*.whl)
[[ ${#wheel_files[@]} -eq 1 && -f "${wheel_files[0]}" ]] \
  || fail "expected exactly one wheel in $wheel_dir, found ${#wheel_files[@]}: ${wheel_files[*]}"
pip install --ignore-installed --no-deps --prefix "$install_root" "${wheel_files[0]}" \
  || fail "pip install of ${wheel_files[0]} into $install_root failed"

site_packages="$(find "$install_root/lib" -type d -name site-packages -print -quit)" \
  || fail "could not search $install_root/lib for a site-packages directory"
[[ -n "$site_packages" ]] \
  || fail "installed wheel did not create a site-packages directory under $install_root/lib"

export PATH="$install_root/bin:$PATH"
export PYTHONPATH="$site_packages${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_dir" || fail "could not enter the scratch project directory $project_dir"

# Read the version back from the installed distribution rather than the wheel
# filename so the assertion covers what an end user's interpreter resolves.
installed_version="$(python -c 'from importlib.metadata import version; print(version("phasesweep"))')" \
  || fail "could not read the installed phasesweep version from $site_packages"

if [[ "$scm_distance" == "0" ]]; then
  # Exactly on a tag: setuptools-scm emits the tag, plus a local segment when
  # the tree is dirty.
  [[ "$installed_version" == "$tag_version" || "$installed_version" == "$tag_version"+* ]] \
    || fail "installed version '$installed_version' does not match tag '$scm_tag' (describe: '$source_describe')"
else
  [[ "$installed_version" =~ ^([^+]+)\+g([0-9a-fA-F]+) ]] \
    || fail "installed version '$installed_version' is not the expected '<release>.dev${scm_distance}+g<node>' shape (describe: '$source_describe')"
  installed_public="${BASH_REMATCH[1]}"
  installed_node="${BASH_REMATCH[2]}"
  [[ "$installed_public" == *".dev${scm_distance}" ]] \
    || fail "installed version '$installed_version' does not carry '.dev${scm_distance}' for the ${scm_distance} commits since $scm_tag (describe: '$source_describe')"
  [[ "$head_hash" == "$installed_node"* ]] \
    || fail "installed version node 'g$installed_node' is not a prefix of HEAD $head_hash (describe: '$source_describe')"
fi

# Named guard for the known setuptools-scm fallback: with unusable git metadata
# it invents 0.1.dev<total commit count> instead of failing the build.
if [[ "$installed_version" == 0.1.dev* && "$tag_version" != 0.1.* ]]; then
  fail "installed version '$installed_version' is the setuptools-scm no-git-metadata fallback; expected a version derived from tag '$scm_tag'"
fi

python -c 'import phasesweep, pathlib, sys; sys.exit(0 if pathlib.Path(phasesweep.__file__).is_relative_to(pathlib.Path(sys.argv[1])) else 1)' "$site_packages" \
  || fail "imported phasesweep from outside $site_packages; the smoke would be testing the checkout, not the wheel"
python -c 'import phasesweep.examples.fake_train' \
  || fail "installed wheel cannot import phasesweep.examples.fake_train"

command -v phasesweep >/dev/null \
  || fail "console script 'phasesweep' is missing from $install_root/bin"
command -v phasesweep-mcp >/dev/null \
  || fail "console script 'phasesweep-mcp' is missing from $install_root/bin"
phasesweep --version | grep -qF "$installed_version" \
  || fail "'phasesweep --version' does not report the installed version $installed_version"

test -f "$site_packages/phasesweep/py.typed" \
  || fail "packaged marker missing: $site_packages/phasesweep/py.typed"
test -f "$site_packages/phasesweep/templates/starter_experiment.yaml" \
  || fail "packaged starter template missing: $site_packages/phasesweep/templates/starter_experiment.yaml"
test -f "$site_packages/phasesweep/mcp/agent_prompt.md" \
  || fail "packaged MCP agent prompt missing: $site_packages/phasesweep/mcp/agent_prompt.md"

phasesweep init || fail "'phasesweep init' failed in $project_dir"
test -f experiment.yaml || fail "'phasesweep init' did not write $project_dir/experiment.yaml"
phasesweep validate experiment.yaml || fail "'phasesweep validate experiment.yaml' failed"
phasesweep run experiment.yaml --dry-run || fail "'phasesweep run experiment.yaml --dry-run' failed"
phasesweep mcp init-catalog --from experiment.yaml -o catalog.yaml \
  || fail "'phasesweep mcp init-catalog' failed"
test -f catalog.yaml || fail "'phasesweep mcp init-catalog' did not write $project_dir/catalog.yaml"

echo "Installed-wheel starter smoke passed (phasesweep $installed_version from $source_describe)."

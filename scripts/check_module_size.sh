#!/usr/bin/env bash
# Refuse any source or test module longer than the line limit. A module past it
# is too large to read and review whole, so it must be split before it changes.
# The test modules listed in `ceilings` already exceeded the limit when it
# reached tests/; each may shrink but never grow past its ceiling. Lower a
# ceiling when its file shrinks, and drop the entry once the file fits the limit.
# With file arguments (the pre-commit hook) it checks those files; without
# any (CI) it checks every tracked module under src/ and tests/.
set -euo pipefail

limit=2000
declare -A ceilings=(
  [tests/test_fingerprint.py]=2097
  [tests/test_mcp_runner.py]=2413
  [tests/test_process_supervision.py]=2214
)

if [ "$#" -eq 0 ]; then
  repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  cd -- "$repo_root"
  mapfile -t modules < <(git ls-files -- 'src/*.py' 'tests/*.py')
  set -- "${modules[@]}"
fi

status=0
for module in "$@"; do
  lines=$(wc -l < "$module")
  allowed="${ceilings[$module]:-$limit}"
  if [ "$lines" -gt "$allowed" ]; then
    echo "check_module_size: $module has $lines lines (limit $allowed); split it" >&2
    status=1
  fi
done
exit "$status"

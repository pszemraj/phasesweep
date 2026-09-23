#!/usr/bin/env bash
# Refuse any source module longer than the line limit. A module past it is too
# large to read and review whole, so it must be split before it changes.
# With file arguments (the pre-commit hook) it checks those files; without
# any (CI) it checks every tracked module under src/.
set -euo pipefail

limit=2000

if [ "$#" -eq 0 ]; then
  repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
  cd -- "$repo_root"
  mapfile -t modules < <(git ls-files -- 'src/*.py')
  set -- "${modules[@]}"
fi

status=0
for module in "$@"; do
  lines=$(wc -l < "$module")
  if [ "$lines" -gt "$limit" ]; then
    echo "check_module_size: $module has $lines lines (limit $limit); split it" >&2
    status=1
  fi
done
exit "$status"

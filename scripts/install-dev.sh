#!/usr/bin/env bash
# Editable install with IDE-friendly paths (Pylance/Cursor autocomplete).
# Uses setuptools "compat" mode so site-packages gets a static .pth file
# instead of a dynamic import hook.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pip install -e . --config-settings editable_mode=compat

echo "Installed sc_utils in editable (compat) mode from $repo_root"
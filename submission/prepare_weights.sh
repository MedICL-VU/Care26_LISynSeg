#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# != 1 ]]; then
  echo "Usage: bash prepare_weights.sh <model_bundle-directory | snapshot-archive.tar.gz>" >&2
  exit 1
fi
source_path="$1"
if [[ -e model_bundle ]]; then
  echo "model_bundle already exists; move it aside before preparing a replacement." >&2
  exit 1
fi

staging=".model_bundle.$$"
container=""
cleanup() {
  [[ -z "$container" ]] || docker rm "$container" >/dev/null
  rm -rf "$staging"
}
trap cleanup EXIT
mkdir "$staging"

if [[ -d "$source_path" ]]; then
  cp -R "$source_path"/. "$staging"/
else
  python3 - "$source_path" <<'PY'
import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], 'rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
expected = '010017368ad11785c0331a6033eccd4cf2f331a9728dba5c727b01604f3f2ff9'
if digest.hexdigest() != expected:
    raise SystemExit('Archive SHA-256 mismatch; use the linked snapshot archive.')
PY
  docker load --input "$source_path"
  container=$(docker create --entrypoint /bin/true care-whs-vmheart-test:submission1-snapshot-20260803)
  docker cp "$container:/models/bundle/." "$staging/"
fi

python3 - "$staging" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads(Path('model_bundle_manifest.json').read_text())
for model in manifest['models']:
    entries = dict(model['metadata_sha256'])
    entries.update({f"fold_{c['fold']}/{c['checkpoint_name']}": c['destination_sha256'] for c in model['checkpoints']})
    for relative, expected in entries.items():
        path = root / model['name'] / relative
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit(f'Weight hash mismatch: {path}')
print('Verified 8 model families and 41 checkpoints.')
PY
mv "$staging" model_bundle

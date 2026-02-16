#!/bin/bash
set -e

# Download ROM and base state files from MinIO at startup
# These are gitignored and can't be baked into the image during OpenShift builds
echo "Downloading ROM files from MinIO..."

python3 -c "
import os, sys
from agent.utils.config import S3Config
from agent.utils.s3 import S3Client

s3 = S3Client(S3Config())
rom_dir = os.environ.get('ROM_DIR', '/home/trainer/roms')
os.makedirs(rom_dir, exist_ok=True)

# Required: ROM file
required = {
    'roms/zelda.gbc': os.path.join(rom_dir, 'zelda.gbc'),
    'roms/zelda.gbc.ram': os.path.join(rom_dir, 'zelda.gbc.ram'),
}

# Optional: download save state if SAVE_STATE_S3_KEY is set
save_key = os.environ.get('SAVE_STATE_S3_KEY', '')
save_path = os.environ.get('SAVE_STATE_PATH', '')
if save_key and save_path:
    required[save_key] = save_path

for s3_key, local_path in required.items():
    if os.path.exists(local_path):
        print(f'  {local_path} already exists, skipping')
        continue
    try:
        data = s3.download_bytes('zelda-models', s3_key)
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(data)
        print(f'  Downloaded {s3_key} -> {local_path} ({len(data)} bytes)')
    except Exception as e:
        print(f'  WARNING: Failed to download {s3_key}: {e}', file=sys.stderr)
        if s3_key == 'roms/zelda.gbc':
            print('  ERROR: ROM file is required, exiting', file=sys.stderr)
            sys.exit(1)

print('ROM files ready.')
"

# Run the training script (or whatever command was passed)
exec "$@"

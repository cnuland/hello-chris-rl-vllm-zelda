#!/bin/bash
set -e

IMAGE_NAME="quay.io/cnuland/zelda-kuberay-worker"
VERSION="${1:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${VERSION}"

echo "Building ${FULL_IMAGE}..."
podman build -t "${FULL_IMAGE}" -f Containerfile .

echo ""
echo "Build complete: ${FULL_IMAGE}"
podman images "${IMAGE_NAME}"

echo ""
read -p "Push to registry? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    podman push "${FULL_IMAGE}"
    echo "Pushed ${FULL_IMAGE}"
fi

"""MinIO/S3 client wrapper for episode and model storage."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from agent.utils.config import S3Config

logger = logging.getLogger(__name__)


class S3Client:
    """Thin wrapper around boto3 for MinIO episode/model storage."""

    def __init__(self, config: S3Config | None = None):
        if config is None:
            config = S3Config()
        self._config = config
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name="us-east-1",
        )

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            self._client.create_bucket(Bucket=bucket)
            logger.info("Created bucket %s", bucket)

    def upload_bytes(self, bucket: str, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=bucket, Key=key, Body=data)

    def upload_json(self, bucket: str, key: str, obj: dict | list) -> None:
        self.upload_bytes(bucket, key, json.dumps(obj).encode())

    def upload_file(self, bucket: str, key: str, path: Path) -> None:
        self._client.upload_file(str(path), bucket, key)

    def download_bytes(self, bucket: str, key: str) -> bytes:
        resp = self._client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()

    def download_json(self, bucket: str, key: str) -> dict | list:
        data = self.download_bytes(bucket, key)
        return json.loads(data)

    def list_keys(self, bucket: str, prefix: str = "", max_keys: int = 0) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
                if max_keys and len(keys) >= max_keys:
                    return keys
        return keys

    def list_manifests(self, bucket: str, prefix: str = "", max_count: int = 0) -> list[str]:
        """List manifest.json keys by walking the two-level directory structure.

        Structure: {episode_id}/{segment_id}/manifest.json
        Uses S3 delimiter listing to avoid scanning all frame objects.
        """
        manifests = []
        # Level 1: list episode-level prefixes
        ep_paginator = self._client.get_paginator("list_objects_v2")
        episode_prefixes = []
        for page in ep_paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                episode_prefixes.append(cp["Prefix"])

        # Level 2: for each episode, list segment-level prefixes
        for ep_prefix in episode_prefixes:
            seg_paginator = self._client.get_paginator("list_objects_v2")
            for page in seg_paginator.paginate(Bucket=bucket, Prefix=ep_prefix, Delimiter="/"):
                for cp in page.get("CommonPrefixes", []):
                    manifests.append(cp["Prefix"] + "manifest.json")
                    if max_count and len(manifests) >= max_count:
                        return manifests

        return manifests

    def delete_all_objects(self, bucket: str, prefix: str = "") -> int:
        """Delete all objects in a bucket (optionally filtered by prefix).

        Returns number of objects deleted.
        """
        deleted = 0
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if objects:
                self._client.delete_objects(
                    Bucket=bucket, Delete={"Objects": objects}
                )
                deleted += len(objects)
        return deleted

    def download_fileobj(self, bucket: str, key: str) -> io.BytesIO:
        buf = io.BytesIO()
        self._client.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return buf

    def force_reset_bucket(self, bucket: str) -> None:
        """Delete all objects and recreate a bucket (faster than paging deletes)."""
        try:
            # Delete all objects in batches
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket):
                objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if objects:
                    self._client.delete_objects(
                        Bucket=bucket, Delete={"Objects": objects}
                    )
            # Delete and recreate bucket
            self._client.delete_bucket(Bucket=bucket)
            self._client.create_bucket(Bucket=bucket)
            logger.info("Reset bucket %s", bucket)
        except ClientError as e:
            if "NoSuchBucket" in str(e):
                self._client.create_bucket(Bucket=bucket)
            else:
                raise

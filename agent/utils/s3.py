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

    def list_keys(self, bucket: str, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def download_fileobj(self, bucket: str, key: str) -> io.BytesIO:
        buf = io.BytesIO()
        self._client.download_fileobj(bucket, key, buf)
        buf.seek(0)
        return buf

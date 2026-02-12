"""Clean all objects from MinIO buckets. Run via Ray job submission."""
import boto3
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

endpoint = os.getenv("S3_ENDPOINT_URL", "http://minio-api.zelda-rl.svc.cluster.local:9000")
s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.getenv("S3_ACCESS_KEY", "admin"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY", "zelda-rl-minio-2024"),
    region_name="us-east-1",
)

for bucket in ["zelda-episodes", "zelda-models"]:
    print(f"Cleaning {bucket}...")
    total = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
            total += len(objects)
            if total % 1000 == 0:
                print(f"  Deleted {total} objects from {bucket}...")
    print(f"  Total deleted from {bucket}: {total}")

print("Cleanup complete!")

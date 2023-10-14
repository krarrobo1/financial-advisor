import boto3
from src.settings import AWS_ACCESS_KEY, AWS_SECRET_KEY

s3 = boto3.resource(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

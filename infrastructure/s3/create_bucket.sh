#!/usr/bin/env bash

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is not set}"
PROJECT_NAME="${PROJECT_NAME:-aws-customer-churn-mlops}"

S3_BUCKET="${PROJECT_NAME}-${AWS_ACCOUNT_ID}-${AWS_REGION}"

echo "Creating S3 bucket:"
echo "  Bucket: $S3_BUCKET"
echo "  Region: $AWS_REGION"

if aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
    echo "Bucket already exists and is accessible."
else
    aws s3api create-bucket \
        --bucket "$S3_BUCKET" \
        --region "$AWS_REGION"

    echo "Bucket created successfully."
fi

echo "Enabling S3 versioning..."

aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled

echo "Enabling default S3 encryption..."

aws s3api put-bucket-encryption \
    --bucket "$S3_BUCKET" \
    --server-side-encryption-configuration \
    '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }'

echo "S3 infrastructure configured successfully."
echo "Bucket: s3://$S3_BUCKET"

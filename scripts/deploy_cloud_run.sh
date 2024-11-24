#!/bin/bash

# Exit the script if any command fails
set -e

# Define variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="mind-sift-app"
REPO_NAME="docker-repo"
REPO_LOCATION="$REGION"

# Ensure Artifact Registry repository exists
echo "Checking if Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REPO_LOCATION" &>/dev/null; then
  echo "Creating Artifact Registry repository..."
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REPO_LOCATION" \
    --description="Docker repository for Cloud Run services"
fi

# Authenticate Docker with Artifact Registry
echo "Authenticating Docker with Artifact Registry..."
gcloud auth configure-docker "$REPO_LOCATION-docker.pkg.dev"

# Build the Docker image
IMAGE_URI="$REPO_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME"
echo "Building Docker image..."
docker build -t "$IMAGE_URI" .

# Push the Docker image to Artifact Registry
echo "Pushing Docker image to Artifact Registry..."
docker push "$IMAGE_URI"

AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
AWS_DEFAULT_REGION=$(aws configure get region)

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_URI" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars LANGCHAIN_TRACING_V2="true" \
  --set-env-vars OPENAI_ORGANIZATION=${OPENAI_ORGANIZATION} \
  --set-env-vars OPENAI_API_KEY=${OPENAI_API_KEY} \
  --set-env-vars LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY} \
  --set-env-vars LANGCHAIN_PROJECT=${LANGCHAIN_PROJECT} \
  --set-env-vars ZILLIZ_CLOUD_URI=${ZILLIZ_CLOUD_URI} \
  --set-env-vars ZILLIZ_CLOUD_USER=${ZILLIZ_CLOUD_USER} \
  --set-env-vars ZILLIZ_CLOUD_PASSWORD=${ZILLIZ_CLOUD_PASSWORD} \
  --set-env-vars AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --set-env-vars AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --set-env-vars AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
  --set-env-vars SUPABASE_URL=${SUPABASE_URL} \
  --set-env-vars SUPABASE_KEY=${SUPABASE_KEY}

echo "Deployment complete! Service is open to the world."

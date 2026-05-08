#!/bin/bash
# ClearML Serving deployment script
# Run this AFTER experiment2.py has completed and registered the model in ClearML.

set -e

echo "=== ClearML Serving Deployment ==="
echo ""

# Defaults can be overridden:
#   CLEARML_SERVING_TASK_ID=<service_id> CLEARML_MODEL_ID=<model_id> bash deploy_serving.sh
SERVICE_ID="${CLEARML_SERVING_TASK_ID:-f4898d5556a942c19f571692a97737a6}"
MODEL_ID="${CLEARML_MODEL_ID:-c69c03966cad4898bcc6cdd8de3d61db}"
CONTAINER_NAME="${CLEARML_SERVING_CONTAINER:-clearml_pm_serving}"
IMAGE="${CLEARML_SERVING_IMAGE:-allegroai/clearml-serving-inference:1.3.2}"
CONF_PATH="${CLEARML_CONF:-$HOME/clearml.conf}"

if [ ! -f "$CONF_PATH" ]; then
  echo "ClearML config not found: $CONF_PATH"
  echo "Run clearml-init first or set CLEARML_CONF=/path/to/clearml.conf"
  exit 1
fi

# Step 1: Install clearml-serving (if not already installed)
pip install clearml-serving

# Step 2: Register model endpoint (already done, kept for reference)
# clearml-serving --id "$SERVICE_ID" model add \
#   --engine sklearn \
#   --endpoint "workers_compensation" \
#   --preprocess "clearml_serving/preprocess.py" \
#   --model-id "$MODEL_ID"

# Step 3: Start inference container
echo "Starting inference container..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  -v "$CONF_PATH:/root/clearml.conf:ro" \
  -p 8080:8080 \
  -e CLEARML_SERVING_TASK_ID="$SERVICE_ID" \
  -e CLEARML_SERVING_POLL_FREQ=5 \
  -e CLEARML_EXTRA_PYTHON_PACKAGES="scikit-learn==1.6.1 xgboost==3.2.0 pandas==2.2.3" \
  "$IMAGE"

echo ""
echo "=== Deployment complete! ==="
echo "Container:          $CONTAINER_NAME"
echo "Docker image:       $IMAGE"
echo "Serving service ID: $SERVICE_ID"
echo "Model ID:           $MODEL_ID"
echo "Inference endpoint: http://localhost:8080/serve/workers_compensation"
echo ""
echo "Test with:"
echo 'curl -X POST "http://localhost:8080/serve/workers_compensation" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"Age": 35, "WeeklyPay": 500, "InitialCaseEstimate": 5000,
  "HoursWorkedPerWeek": 40, "DaysWorkedPerWeek": 5,
  "Gender": "M", "MaritalStatus": "S",
  "DependentChildren": 0, "DependentsOther": 0,
  "PartTimeFullTime": "F",
  "Accident_Year": 2010, "Accident_Month": 6, "Accident_DayOfWeek": 2,
  "Reported_Year": 2010, "Reported_Month": 7, "Reported_DayOfWeek": 1,
  "ReportDelay_Days": 30}'"'"

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# docker-entrypoint.sh
# Controls which script to run based on the command passed to docker run
#
# Usage:
#   docker run apex-propensity-model generate   → runs generate_dataset.py
#   docker run apex-propensity-model train      → runs train_model.py
#   docker run apex-propensity-model evaluate   → runs evaluate_model.py
#   docker run apex-propensity-model predict    → runs predict.py
#   docker run apex-propensity-model all        → runs full pipeline
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit immediately if any command fails

COMMAND=${1:-all}

echo "============================================================"
echo "  Apex Securities — Propensity Model Container"
echo "  Command: $COMMAND"
echo "============================================================"

case "$COMMAND" in

    generate)
        echo "Running: Dataset Generation"
        python generate_dataset.py
        ;;

    train)
        echo "Running: Model Training"
        python train_model.py
        ;;

    evaluate)
        echo "Running: Model Evaluation"
        python evaluate_model.py
        ;;

    predict)
        echo "Running: Customer Scoring"
        python predict.py
        ;;

    all)
        echo "Running: Full Pipeline (generate → train → evaluate → predict)"
        echo ""
        echo "--- Step 1: Generating Dataset ---"
        python generate_dataset.py
        echo ""
        echo "--- Step 2: Training Models ---"
        python train_model.py
        echo ""
        echo "--- Step 3: Evaluating Models ---"
        python evaluate_model.py
        echo ""
        echo "--- Step 4: Scoring New Customers ---"
        python predict.py
        echo ""
        echo "Full pipeline complete!"
        ;;

    *)
        echo "Unknown command: $COMMAND"
        echo "Valid commands: generate | train | evaluate | predict | all"
        exit 1
        ;;
esac

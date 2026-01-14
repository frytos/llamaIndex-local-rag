#!/bin/bash
# ============================================================================
# RunPod Automatic Initialization Script
# ============================================================================
# This script runs automatically when the pod starts.
# It sets up the RAG pipeline environment in the background.
#
# Usage: This is called automatically via docker_args on pod creation
# ============================================================================

set -e

echo "=========================================================================="
echo "RunPod Auto-Init - Starting Background Setup"
echo "=========================================================================="
echo "Pod will be available for SSH immediately."
echo "Services will be initialized in the background (~5-10 minutes)."
echo ""

# Create log directory
mkdir -p /workspace/rag-pipeline/logs

# Run initialization in background
(
    cd /workspace

    # Log file
    LOG_FILE="/workspace/rag-pipeline/logs/auto-init.log"

    echo "===========================================================================" > $LOG_FILE
    echo "Auto-Init Started: $(date)" >> $LOG_FILE
    echo "===========================================================================" >> $LOG_FILE
    echo "" >> $LOG_FILE

    # Wait a moment for SSH to be fully ready
    sleep 5

    # Check if init script exists
    if [ ! -f "/workspace/rag-pipeline/scripts/init_runpod_services.sh" ]; then
        echo "⚠️  Init script not found. Waiting for it to be uploaded..." >> $LOG_FILE
        echo "   Expected: /workspace/rag-pipeline/scripts/init_runpod_services.sh" >> $LOG_FILE
        echo "" >> $LOG_FILE
        echo "📝 To complete setup, upload your repository:" >> $LOG_FILE
        echo "   scp -r /path/to/llamaIndex-local-rag POD_HOST@ssh.runpod.io:/workspace/rag-pipeline" >> $LOG_FILE
        echo "   Then SSH in and run: bash /workspace/rag-pipeline/scripts/init_runpod_services.sh" >> $LOG_FILE
    else
        echo "✅ Init script found! Running automated setup..." >> $LOG_FILE
        echo "" >> $LOG_FILE

        # Run the init script
        bash /workspace/rag-pipeline/scripts/init_runpod_services.sh >> $LOG_FILE 2>&1

        if [ $? -eq 0 ]; then
            echo "" >> $LOG_FILE
            echo "===========================================================================" >> $LOG_FILE
            echo "✅ AUTO-INIT COMPLETE: $(date)" >> $LOG_FILE
            echo "===========================================================================" >> $LOG_FILE
            echo "Services are ready!" >> $LOG_FILE
            echo "" >> $LOG_FILE
            echo "Test with:" >> $LOG_FILE
            echo "  curl http://localhost:8000/health" >> $LOG_FILE
            echo "  psql -h localhost -U fryt -d vector_db -c 'SELECT 1'" >> $LOG_FILE
        else
            echo "" >> $LOG_FILE
            echo "❌ AUTO-INIT FAILED: $(date)" >> $LOG_FILE
            echo "Check logs above for errors" >> $LOG_FILE
        fi
    fi

    echo "" >> $LOG_FILE
    echo "===========================================================================" >> $LOG_FILE
    echo "Auto-Init Process Finished: $(date)" >> $LOG_FILE
    echo "===========================================================================" >> $LOG_FILE

) &

# Create a status file
cat > /workspace/rag-pipeline/STATUS.txt << 'EOF'
========================================================================
RAG PIPELINE POD STATUS
========================================================================

Pod is READY for SSH access!

Background initialization is in progress...

Check progress:
  tail -f /workspace/rag-pipeline/logs/auto-init.log

Manual setup (if needed):
  1. Upload your code:
     scp -r /path/to/llamaIndex-local-rag POD_HOST@ssh.runpod.io:/workspace/rag-pipeline

  2. Run init script:
     bash /workspace/rag-pipeline/scripts/init_runpod_services.sh

Services will be available when auto-init completes (~5-10 minutes).

========================================================================
EOF

cat /workspace/rag-pipeline/STATUS.txt

# Keep container running
exec sleep infinity

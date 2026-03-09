#!/bin/bash
# Quick file transfer to/from RunPod
# Usage:
#   ./pod.sh flux connect                  # SSH into flux pod
#   ./pod.sh sdxl upload <local> <remote>  # Upload to sdxl pod
#   ./pod.sh flux loras                    # Download flux LoRA checkpoints
#   ./pod.sh sdxl loras                    # Download sdxl LoRA checkpoints

# ── Pod configs ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/pod_config.sh" ]]; then
  source "$SCRIPT_DIR/pod_config.sh"
else
  echo "No pod_config.sh found. Copy pod_config.example.sh and fill in your pod IPs."
  exit 1
fi
# ──────────────────────────────────────────────

POD="$1"
CMD="$2"
shift 2 2>/dev/null

# Resolve pod name to config vars
case "$POD" in
  flux) IP="$FLUX_IP"; PORT="$FLUX_PORT"; JOB="$FLUX_JOB" ;;
  sdxl) IP="$SDXL_IP"; PORT="$SDXL_PORT"; JOB="$SDXL_JOB" ;;
  *)
    echo "Usage: ./pod.sh <flux|sdxl> <command> [args...]"
    echo ""
    echo "Pods:"
    echo "  flux   FLUX training (A100 80GB)"
    echo "  sdxl   SDXL retrain (A40 48GB)"
    echo ""
    echo "Commands:"
    echo "  connect                    SSH into pod"
    echo "  upload <local> <remote>    Upload to pod"
    echo "  download <remote> <local>  Download from pod"
    echo "  loras [job]                Download LoRA checkpoints"
    echo "  samples [job]              Download training samples"
    echo "  stop                       Stop the pod"
    exit 1
    ;;
esac

SCP_OPTS="-o StrictHostKeyChecking=no -P $PORT"
SSH_OPTS="-o StrictHostKeyChecking=no -p $PORT"

case "$CMD" in
  connect)
    ssh $SSH_OPTS root@$IP
    ;;
  upload)
    scp -r $SCP_OPTS "$1" "root@$IP:${2:-/workspace/}"
    ;;
  download)
    scp -r $SCP_OPTS "root@$IP:$1" "${2:-.}"
    ;;
  loras)
    J="${1:-$JOB}"
    mkdir -p loras
    scp $SCP_OPTS "root@$IP:/workspace/output/$J/$J/*.safetensors" loras/
    echo "Downloaded to loras/"
    ls -lh loras/*.safetensors 2>/dev/null
    ;;
  samples)
    J="${1:-$JOB}"
    mkdir -p outputs/samples_${POD}
    scp -r $SCP_OPTS "root@$IP:/workspace/output/$J/$J/samples/" outputs/samples_${POD}/
    echo "Downloaded to outputs/samples_${POD}/"
    ;;
  stop)
    ssh $SSH_OPTS root@$IP "runpodctl stop pod" 2>/dev/null
    echo "Pod $POD stop requested"
    ;;
  *)
    echo "Unknown command: $CMD"
    echo "Commands: connect, upload, download, loras, samples, stop"
    ;;
esac

#!/bin/bash
# Quick file transfer to/from RunPod
# Usage:
#   ./pod.sh flux connect                  # SSH into flux pod
#   ./pod.sh sdxl upload <local> <remote>  # Upload to sdxl pod
#   ./pod.sh flux loras                    # Download flux LoRA checkpoints
#   ./pod.sh sdxl loras                    # Download sdxl LoRA checkpoints

# ── Pod configs ───────────────────────────────
declare -A POD_IPS POD_PORTS POD_JOBS
POD_IPS[flux]="64.247.206.116"
POD_PORTS[flux]="17763"
POD_JOBS[flux]="sim_aesthetic_flux"

POD_IPS[sdxl]="195.26.232.162"
POD_PORTS[sdxl]="56746"
POD_JOBS[sdxl]="sim_aesthetic_sdxl_v2"
# ──────────────────────────────────────────────

POD="$1"
CMD="$2"
shift 2 2>/dev/null

if [[ -z "${POD_IPS[$POD]}" ]]; then
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
fi

IP="${POD_IPS[$POD]}"
PORT="${POD_PORTS[$POD]}"
JOB="${POD_JOBS[$POD]}"
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

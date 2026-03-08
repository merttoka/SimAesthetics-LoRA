#!/bin/bash
# Quick file transfer to/from RunPod
# Usage:
#   ./pod.sh connect                     # SSH into pod
#   ./pod.sh upload <local> <remote>     # Upload file/dir to pod
#   ./pod.sh download <remote> <local>   # Download file/dir from pod
#   ./pod.sh loras                       # Download all LoRA checkpoints
#   ./pod.sh samples                     # Download sample images

# ── Config (update these) ──────────────────
POD_IP="195.26.232.162"
POD_PORT="37626"
# ────────────────────────────────────────────

SCP_OPTS="-o StrictHostKeyChecking=no -P $POD_PORT"
SSH_OPTS="-o StrictHostKeyChecking=no -p $POD_PORT"

case "$1" in
  connect)
    ssh $SSH_OPTS root@$POD_IP
    ;;
  upload)
    scp -r $SCP_OPTS "$2" "root@$POD_IP:${3:-/workspace/}"
    ;;
  download)
    scp -r $SCP_OPTS "root@$POD_IP:$2" "${3:-.}"
    ;;
  loras)
    JOB="${2:-sim_aesthetic_sdxl}"
    mkdir -p loras
    scp $SCP_OPTS "root@$POD_IP:/workspace/output/$JOB/$JOB/*.safetensors" loras/
    echo "Downloaded to loras/"
    ls -lh loras/*.safetensors 2>/dev/null
    ;;
  samples)
    JOB="${2:-sim_aesthetic_sdxl}"
    mkdir -p outputs/samples
    scp -r $SCP_OPTS "root@$POD_IP:/workspace/output/$JOB/$JOB/samples/" outputs/samples/
    echo "Downloaded to outputs/samples/"
    ;;
  stop)
    ssh $SSH_OPTS root@$POD_IP "runpodctl stop pod" 2>/dev/null
    echo "Pod stop requested"
    ;;
  *)
    echo "Usage: ./pod.sh {connect|upload|download|loras|samples|stop}"
    echo "  connect                    SSH into pod"
    echo "  upload <local> <remote>    Upload to pod"
    echo "  download <remote> <local>  Download from pod"
    echo "  loras                      Download all LoRA checkpoints"
    echo "  samples                    Download training sample images"
    echo "  stop                       Stop the pod"
    ;;
esac

docker run -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --shm-size 32G \
  -e TZ=Asia/Tokyo \
  -w /workspace neumann_tim \
  "$@"
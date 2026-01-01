docker run -it \
  --gpus '"device=4"' \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --shm-size 32G \
  -e TZ=Asia/Tokyo \
  -w /workspace \
  -v "$(pwd)":/workspace \
  neumann_tim

  # bash -c "pip install -e . && exec \"\$@\"" _ \
  # "$@"
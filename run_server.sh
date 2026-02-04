export ATOM_TORCH_PROFILER_DIR=/root
# python3 -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --server-port 8888 --kv_cache_dtype fp8 --block-size 16
python3 -m atom.entrypoints.openai_server --model /root/DeepSeek-R1-0528 --server-port 8888 -tp 8 --kv_cache_dtype fp8 --block-size 16

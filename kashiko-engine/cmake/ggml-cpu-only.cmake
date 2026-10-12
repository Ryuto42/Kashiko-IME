# Included before every project() of the llama.cpp build (see .cargo/config.toml).
# Inference runs on the CPU (n_gpu_layers = 0), so the Metal backend is never
# used — but when compiled in, it still initializes at startup and compiles its
# shaders, which costs seconds of CPU and memory for nothing.
set(GGML_METAL OFF CACHE BOOL "" FORCE)

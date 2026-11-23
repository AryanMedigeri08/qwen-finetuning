# CPU inference with llama.cpp

The direct Transformers runtime uses the original adapter. On a machine without CUDA, using the optional bitsandbytes dependency and setting SAHAYAK_NF4=1 lowers its memory requirement. The CPU NF4 run still takes longer than the T4 training notebook's probes.

For local interaction and exploratory CPU comparisons, llama.cpp can apply a converted copy of the same LoRA adapter to a GGUF copy of the base model. Record Q4_K_M results separately from the original NF4 GPU runs.

## Prepare the files

Use the official [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12651) and the corresponding source checkout. Point the converters to the downloaded Qwen snapshot identified in sahayak/model.py and the restored C2 adapter directory. Run the commands with the working directory set to the SahayakLLM checkout.

    python <llama-source>/convert_hf_to_gguf.py <qwen-snapshot> --outtype f16 --outfile .runtime/qwen-f16.gguf
    python <llama-source>/convert_lora_to_gguf.py --base <qwen-snapshot> --outtype f16 --outfile .runtime/sahayak-adapter.gguf .runtime/C2_adapter
    <llama-bin>/llama-quantize .runtime/qwen-f16.gguf .runtime/qwen-q4_k_m.gguf Q4_K_M 6
    <llama-bin>/llama-server -m .runtime/qwen-q4_k_m.gguf --lora .runtime/sahayak-adapter.gguf --host 127.0.0.1 --port 8098 --parallel 1 --ctx-size 2048 --threads 6

Use the matching Windows executables with the .exe suffix. The converter uses its own dependency file. The server binds to the local machine and applies the adapter at scale 1 for C2, C4 and C5 and scale 0 for C0, C1 and C3.

## Run the app and benchmark

Set SAHAYAK_LOCAL_SERVER=http://127.0.0.1:8098 before starting Streamlit to use the local server. Keep the server open while the app runs.

    python -m sahayak.benchmark --local-server http://127.0.0.1:8098 --configs C2 C4 C5 --resume --output results/runs/cpu-q4

The benchmark output records its inference engine and precision. Use a separate output directory for the matched T4 experiment.

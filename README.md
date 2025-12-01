# ComfyUI-DGXSparkSafetensorsLoader
**A ComfyUI model loader that uses the fastsafetensors library to perform a very fast, zero-copy load from storage to VRAM.**

_This is very experimental, and may destroy the universe. So please don't use it in a production environment under any circumstances._

On DGX Spark, fastsafetensors is a massive improvement over the Hugging Face safetensors library for loading AI models. The Hugging Face library doesn't work well with the DGX Spark due to its architecture and memory design. Models load very slowly and sometimes use up to 2x memory during loading. This can cause large models to exceed the Spark's RAM capacity and fail, even when the model should fit in under half of the machine's RAM capacity.

This node doesn't require ComfyUI to be launched with the --cache-none or --disable-mmap options. The default options should work fine.

Here's an example of memory usage during and after loading the 60GB FLUX.2-dev BF16 model to the GPU and the 17GB Mistral FP8 text encoder to the CPU. As you can see, model loading happens extremely fast and memory usage never goes over 60%:

![FLUX.2-dev memory usage](https://github.com/user-attachments/assets/6a3a4ff7-bc4e-47ea-99b7-d4961b505a01)

fastsafetensors library project on GitHub:
https://github.com/foundation-model-stack/fastsafetensors

# Known Issues
This currently doesn't seem to work with machines that have discrete GPUs, although the loading problem isn't as bad on systems without shared memory. In this case, just use the normal model loader included with ComfyUI.

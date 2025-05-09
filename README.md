# CS598-VLC

## Overview

- `WhatsUp_test`: Fine-grained benchmarking on the [What's Up dataset](https://arxiv.org/pdf/2310.19785)
- `benchmark`: Benchmarking SOTA models, pre-trained VL models, and our fine-tuned models on a variety of compositional tasks (Winoground, ColorSwap, ARO etc.). Also contains experiments probing the embedding space of [DAC-SAM](https://arxiv.org/pdf/2305.19595). To run code in this folder, follow these steps:  
    - Activate mamba: `module load miniconda`
    - Place yourself in the `benchmark` folder
    - Activate conda env. It is stored in the project folder so we all have access: `mamba activate [project folder]/.conda/envs/dac`
    - Run `python3 -m code.eval`
- `finetuning`: Download Open Images dataset, generate positive captions with BLIP-2-6.7b and negative captions with Mistral- 7B-Instruct, and LoRA finetuning OpenAI ViT-B-32 base model.
- `style_transfer_experiments`: Reproducing style transfer methods from [this paper](https://arxiv.org/pdf/2303.17590). We also experiment with several image prompt adapter models to improve the style transfer results. 



## Notes
- Project folder on SCC is `/projectnb/cs598/projects/comp_reason/CS598-VLC`
- Large files that can not be comitted to github (data and model checkpoints) are located in the project folder



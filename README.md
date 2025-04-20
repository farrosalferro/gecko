# Gecko: Guidance from Keyword for Identifying Small Details in High-Resolution Images with Vision-Language Models

[[Project Page](https://farrosalferro.github.io/gecko.github.io/)] [[Benchmark](https://huggingface.co/datasets/farrosalferro24/multi_vstar_bench)] [[Model Zoo](https://huggingface.co/collections/farrosalferro24/gecko-66d9596ead2d8eb9febe2a0f)]

## Install
1. Clone this repository and navigate to `gecko` folder
```bash
git clone https://github.com/farrosalferro/gecko.git
cd gecko
```

2. Set up the environment and install the required packages:
```bash
conda create -n gecko python=3.10 -y
conda activate gecko
pip install --upgrade pip
pip install -e .
pip install flash-attn==2.5.8 --no-build-isolation
```

## Inference
### Run from Terminal
We provide an inference script in inference.sh. Replace the necessary arguments with your own, then run:
```bash
bash inference.sh
```

### Run with Gradio (Interactive)
For interactive inference, run:
```bash
python run.py
```
This will launch a Gradio interface. Click the generated URL in your terminal to start.

## Evaluation
We provide scripts to evaluate our model on two related benchmarks:
* [vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench)
* [multi_vstar_bench](https://huggingface.co/datasets/farrosalferro24/multi_vstar_bench)

### Setup
First, download the benchmarks and place them inside a benchmarks directory:
```bash
mkdir benchmarks
cd benchmarks
git lfs install
git clone https://huggingface.co/datasets/farrosalferro24/multi_vstar_bench
git clone https://huggingface.co/datasets/craigwu/vstar_bench
cd .. # Return to the root directory
```
Your directory should look like this

```
gecko/
├── examples/
├── model/
├── eval/
└── benchmarks/
    ├── vstar_bench/
    └── multi_vstar_bench/
```

### Run Evaluation

Use the corresponding script for each benchmark:
```bash
cd eval
bash multi_vstar_bench.sh # for multi vstar bench
bash vstar_bench.sh # for vstar bench
```
You can adjust the model settings inside each bash script.

## Citation
If you find this work useful in your research or applications, please cite us with the following BibTeX:
```bibtext
@misc{farros2024gecko,
    title={gecko: Guidance from Keyword for Identifying Small Details in High-Resolution Images with Vision-Language Models},
    url={https://github.com/farrosalferro/gecko.git},
    author={Alferro, Farros and Suganuma, Masanori, and Okatani, Takayuki},
    month={August},
    year={2024}
}
```

## Acknowledgement
- [Mantis](https://arxiv.org/abs/2405.01483v1): The foundational codebase and base model for our plug-and-play module.
- [V*](https://arxiv.org/abs/2312.14135): For proposing the task and benchmark.

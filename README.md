<div align="center">
<h1 align="center">
  <sub>
    <img
      src="https://github.com/bin123apple/GUI_Spotlight/blob/main/asset/logo.png"
      alt="GUI Spotlight Logo"
      width="40"
    />
  </sub>
  GUI-Spotlight: Adaptive Iterative Focus Refinement for Enhanced GUI Visual Grounding.
</h1>

<a href="https://arxiv.org/abs/2510.04039">
  <img
    src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white&style=for-the-badge"
    alt="Paper"
  />
</a>

<a href="https://huggingface.co/Bin12345/GUI-Spotlight">
  <img
    src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=white&style=for-the-badge"
    alt="Hugging Face Model"
  />
</a>

<a href="https://huggingface.co/datasets/nuoxu1993/VG-RL-filter-dataset-hf">
  <img
    src="https://img.shields.io/badge/HuggingFace-Dataset-orange?logo=huggingface&logoColor=white&style=for-the-badge"
    alt="Hugging Face Dataset"
  />
</a>

</div>

## Introduction
GUI_Spotlight is a `think-with-image` GUI visual grounding model. For each step, it first calls tooling to crop the image according to its own predictions, and then returns an exact coordinate location.

## Setup

```
cd GUI_Spotlight
conda create --name spotlight python=3.12
conda activate spotlight
conda install -c conda-forge uv
uv pip install -e .
```

## Evaluation

**Screenspot-pro**
```
python screenspot_pro_evaluation.py
```

**OSWorld-G** (Need to download the dataset by yourself)
```
python osworld_g_evaluation.py \
  --model Bin12345/GUI-Spotlight \
  --dataset_json OSWorld-G_refined.json \
  --images_dir OSWorld-G/benchmark/images \
  --batch_size 1
```

**UI-Vision** (Need to download the dataset by yourself)
```
python uivision_evaluation.py \
  --model Bin12345/GUI-Spotlight \
  --dataset_json `uivision/annotations` \
  --images_dir `ui-vision/images` \
  --batch_size 1
```

## Single Sample Inference

```
python inference.py --prompt `Your prompt` --image_path `Image Path` --model `The name of the model`
```
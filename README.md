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

</div>

## Introduction
GUI_Spotlight is a `think-with-image` model. For each step, it first calls tooling to crop the image according to its own predictions, and then returns an exact coordinate location.

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
  --model  \
  --dataset_json OSWorld-G_refined.json \
  --images_dir OSWorld-G/benchmark/images \
  --batch_size 1
```

**UI-Vision** (Need to download the dataset by yourself)
```
python uivision_evaluation.py \
  --model  \
  --dataset_json `uivision/annotations` \
  --images_dir `ui-vision/images` \
  --batch_size 1
```

## Single Sample Inference

```
python inference.py --prompt `Your prompt` --image_path `Image Path` --model `The name of the model`
```
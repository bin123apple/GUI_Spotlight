# GUI_Spotlight

## Setup

```
cd GUI_Spotlight
conda create --name spotlight python=3.12
conda activate spotlight
conda install -c conda-forge uv
uv pip install -e .
```

## Evaluation

```
python screenspot_pro_evaluation.py
```

## Inference

```
python inference.py --prompt `Your prompt` --image_path `Image Path` --model `The name of the model`
```
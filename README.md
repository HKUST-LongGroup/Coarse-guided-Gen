# Coarse-Guided Visual Generation via Weighted $h$-Transform Sampling

## Abstract
We propose a novel guided method by using the $h$-transform, a tool that can constrain the sampling process under desired conditions. Specifically, we modify the transition probability at each sampling timestep by adding to the original differential equation with a drift function, which approximately steers the generation toward the ideal fine sample. To address unavoidable approximation errors, we introduce a noise-level-aware schedule that gradually de-weights the term as the error increases, ensuring both guidance adherence and high-quality synthesis.

## 1. Environment preparation
```
pip install -r requirements.txt
```

## 2.Quick start

### Coarse image guided generation

We take the image super-resolution task and generate the images for the given eight example images.

```
cd ImageGen
```
Download pretrained checkpoint from the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the checkpoint "ffhq_10m.pt" and paste it to ./models/. Then run the code.
```
bash run.sh
```
The result will be stored in the ''VideoGen/outputs/'' path.

### Coarse video guided generation

We generate the video for the given example video.
```
cd ImageGen 
bash run.sh
```
The result will be stored in the ''ImageGen/outputs/'' path.

## 3.Fully run

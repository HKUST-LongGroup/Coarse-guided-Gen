## Environment preparation
```
pip install -r requirements.txt
```

---

## Access Permission of CogVideoX and Wan2.2 on Huggingface
```
pip install --upgrade huggingface_hub
hf auth login
```
Enter your access token.

---


## Quick start for image restoration

We take the image super-resolution task and use eight images as the example.
```
cd VideoGen
```
Download the pretrained ffhq_10m.pt from the internet and put it into the "models/" path
```
bash run.sh
```
The result will be stored in the ''VideoGen/outputs/'' path.

---

## Quick start for video generation
#### We take use one video as the example.
```
cd ImageGen 
bash run.sh
```
The result will be stored in the ''ImageGen/outputs/'' path.
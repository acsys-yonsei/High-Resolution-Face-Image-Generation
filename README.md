# High-Resolution-Face-Image-Generation

## GANformer
## SRGAN
SRGAN can upscale given images while revise quality of them.

### Pretrained Weights
If there is not enough time to train SRGAN from scratch, a pretrained model is needed.      
Pretrained weights are provided below.

[BaiduNetDisk](https://pan.baidu.com/s/15_vhGQdkHIfLCRgo7xanpg): Extraction code：cxp0       
[YandexDisk](https://yadi.sk/d/Pl_hxVZPa_PHew)


After download the file, unzip and put them under `Face-Renovation/checkpoints`.

### Target Data
To set path of source data, open `Face-Renovation/options/config_hifacegan.py` and change variable `dataroot` of `TestOptions` class.       
Also, variable `name` should also be changed to the target model you want to use.       
Some other variables may also be needed to change depending on the target model.

### Inference
```
cd Face-Renovation
python test_nogt.py
```

## Latent Space Manipulation
Manipulating latent vectors in high dimensional latent space allows us to further enhance the quality of GAN-generated images.      
The end-to-end process is as follow:    
First, we train SVM to learn the boundary between "good" and "bad" quality, where the "good" quality image are closer to our desired results from GAN.     
Then, we project latent vectors in the orthogonal direction to the boundary to enhance their quality. 

### Prepare Data
Training SVM requires a dataset that consists of GAN-generated images, scored according to their quality.
We used [GANformer](https://github.com/dorarad/gansformer) to generate images, but any trained GAN is possible.  

For scoring, any single metric value is possible, but our experiment uses Discriminator logits as quality metric. 
(Note that the Discriminator is trained to distinguish between real and fake images, which aligns with our definition of "good" and "bad" quality images.)    
Latent vectors and their scores should be saved as separate files.

Our code for training boundary and inteprolation are taken and modified from the original repo [InterFaceGAN](https://github.com/genforce/interfacegan). 

### Train SVM
Train SVM using saved latent vectors and their scores. Specify output file accordingly.       

    python train_boundary.py \
        -o boundaries/ffhq1024_"$ATTRIBUTE_NAME" \
        -c data/ffhq1024/z.npy \
        -s data/ffhq1024/"$ATTRIBUTE_NAME"_scores.npy

### Interpolate Latent Vector
    python edit.py \
        -m ganformer \
        -o test \
        -b boundaries/ganformer_ffhq1024_quality/boundary.npy \
        -i data/ffhq1024/latents_all.npy \
        -n 100 --max 1000

## gMLP

# ResNet-GAN
This is a torch implementation of the Resnet generator with spectral norm discriminator. 

You can learn more about the Resnet generators here (https://arxiv.org/pdf/1704.00028.pdf)

You can learn more about the spectral norm discriminators here (https://arxiv.org/pdf/1802.05957.pdf)

# Resnet-GAN Archeticture
The main building block in both generators and discriminators is the Res-block. Its archeticture is shown in the following image

![image](https://user-images.githubusercontent.com/47930821/130739125-ec55a98c-29e2-4551-9a2a-b3b18015fa8d.png)

---
# Generator
The generator consists of stack of residual layers to upsample the latent input as shown in the image

![image](https://user-images.githubusercontent.com/47930821/130739011-d7beadb5-bca0-4f25-9924-1d657e929815.png)

# Discriminator
The discriminator consists also of stack of residual layers but it downsample the image and then a dense layer is added to judge the realisticity of the image. The archeticture of the discriminator is shown in the following image

![image](https://user-images.githubusercontent.com/47930821/130739428-61ee148e-96eb-456a-bc7d-b628d7865b5d.png)

# Spectral-Norm penalization
In the discriminator, the spectral norm penalization is applied. Usually in deep learning, it is harmful to let the model lean on a small subset of weights. Therefore weight regularization is sometimes essential. However directly penalizing the weights may restrict the learning process. Thus, instead of penalizing weights, the highest eigen value of the weights is penalized instead. Thus, it prevents the space of the weight matrix to be oriented in one specific direction.

Pros: it helps stabilize the training, since the over-trained discriminator makes the generator diverge during the training

Cons: it makes the training slower

---

# FID score (frechet inception distance)
For assessing the quality of the generative models, this repo used FID score. This metric measures the distance between the InceptionV3 convolutional features' distribution between real and fake images. This metric has been widely used to see how far both distributions are. Therefore, the lower this metric, the better.


You can learn more about FID metric here (https://arxiv.org/abs/1706.08500)

---
# Prerequisites
1- python3 

2- CPU or NVIDIA GPU (GPU is recommended for faster inversion)

---

# Training
Run this script on a colab notebook to start the training. (GPU is required).
Upload this folder in a specified location and write its path in dataset_path: https://drive.google.com/drive/folders/1Fl3UlXUyZ5PhLl_31nQ7kTredhvmi8uQ?usp=sharing
```python
!pip install argparse
!git clone https://github.com/Mohanned-Elkholy/ResNet-GAN
%cd /content/ResNet-GAN
!python main.py --dataset_path "put the path of the dataset folder here" --GAN_check_point_path '/' --outputs '/content/drive/MyDrive/For_Jacob/Githubs/Resnet_GAN/outputs' --continue_training_from_checkpoint False

```
You can also run the colab notebook provided in the repo, or you can open this link: https://colab.research.google.com/drive/1rX0yPS4Vk_XgiIAkhTbbdXSRGZjpwiGa.

---
---


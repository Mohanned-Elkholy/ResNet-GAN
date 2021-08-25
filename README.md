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

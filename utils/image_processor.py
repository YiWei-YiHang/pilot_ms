#TODO 该代码已完成，已检验：preprocess_image,tensor2PIL,待检验：mask4image
import numpy as np
from PIL import Image
import mindspore as ms

def preprocess_image(image):
    """PIL->Tensor"""
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)#增加维度，重排顺序
    image = ms.from_numpy(image).clamp(0,1)
    return 2 * image - 1 #归一化到 [-1, 1] 的主要目的是帮助神经网络训练时更稳定、更快速，并更好地与常见的激活函数配合。

def tensor2PIL(image):
    image = image.squeeze(0)
    image = ((image + 1.0) / 2.0 * 255.0).permute(1,2,0).move_to("CPU").numpy().astype(int)
    image = Image.fromarray(np.uint8(image))
    return image

def mask4image(image, mask):
    image = (image + 1) / 2
    mask = (mask + 1) / 2
    image = image * (mask > 0.5)
    image = 2 * image - 1
    return image



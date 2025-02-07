#TODO:代码已完成,已检验：whitemask4image；待检验：t2i_visualize
from PIL import Image
from pilot_ms.utils.image_processor import preprocess_image, tensor2PIL

def whitemask4image(image,mask_tensor):
    """Return:PIL"""
    mask_add = mask_tensor.clone()
    for i in range(3):
        mask_add[:, i, ...][mask_add[:, i, ...] == 0] = 0.99
        mask_add[:, i, ...][mask_add[:, i, ...] == 1] = 0
        mask_tensor[:, i, ...][mask_tensor[:, i, ...] == 0] = 0
    image_mask  = (preprocess_image(image) + 1) / 2 * mask_tensor#将原图的mask位置变为黑色
    image_mask = image_mask + mask_add#mask位置加上whitemask
    image_mask = 2 * image_mask - 1
    image_mask = tensor2PIL(image_mask)
    return image_mask

def t2i_visualize(image,mask_image,result_list,W=512,H=512):
    mask_tensor = (preprocess_image(mask_image)+1)/2
    image_mask = whitemask4image(image,mask_tensor)
    new_image_list = []
    for i in range(len(result_list)):
        new_width = W * 2
        new_height = H
        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image_mask, (0, 0))
        new_image.paste(result_list[i], (W, 0))
        new_image_list.append(new_image)
    return new_image_list
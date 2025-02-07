from utils.image_processor import preprocess_image, tensor2PIL, mask4image
from utils.visualize import whitemask4image
from PIL import Image

image = Image.open("data/vase-sample.png").convert("RGB")
image = image.resize((512,512), Image.NEAREST)

mask_image = Image.open("data/vase-mask.png").convert("RGB")
mask_image = mask_image.resize((512,512), Image.NEAREST)

mask_tensor = (preprocess_image(mask_image) + 1) / 2
image_mask = whitemask4image(image,mask_tensor)
image_mask.save("outputs/whitemask4image.png")






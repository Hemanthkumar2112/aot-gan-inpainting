import torch
import aotgan
import argparse
import numpy as np
from PIL import Image
import torch
import cv2
from torchvision.transforms import ToTensor
model_path=r"aot_gan_inpaint.pt"
args = argparse.Namespace()
args.block_num = 8
args.rates = [1, 2, 4, 8]

model = aotgan.InpaintGenerator(args)
model.load_state_dict(torch.load(model_path, map_location='cpu'))


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


def infer(img, mask):
    with torch.no_grad():
        img_cv = np.array(img)[:, :, :3]
        # img_cv = Image.fromarray(img_cv).resize((512,512))
        # im_cv = np.asarray(img_cv)
        # Fixing everything to 512 x 512 for this demo.
        img_tensor = (ToTensor()(img_cv) * 2.0 - 1.0).unsqueeze(0)
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)


        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))
        comp_np = postprocess(comp_tensor[0])

        return comp_np


im_path=r"C:\Users\Admin\Downloads\image.jpg"
mas_path=r"C:\Users\Admin\Downloads\mask.png"

img=cv2.resize(cv2.imread(im_path),(512,512))
mask=cv2.resize(cv2.imread(mas_path),(512,512))
mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
comp_np=infer(img, mask)
cv2.imwrite("result.png",comp_np)
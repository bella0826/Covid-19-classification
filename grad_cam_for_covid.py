import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    transform = transforms.Compose({
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    })
    img = transform(img)
    img = img.unsqueeze(0)
    return img

def backward_hook(module, grad_in, grad_out):
    print("hi")
    grad_block.append(grad_out[0].detach())
def forward_hook(module, input, output):
    print("hi")
    fmap_block.append(output)

def cam_show_img(img, featrue_map, grads, our_dir):
    H, W, _ = img.shape
    cam = np.zeros(featrue_map.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0],-1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * featrue_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2. resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img
    path_cam_img = os.path.join(our_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)

if __name__ == '__main__':
    path_img = "./Covid/COVID/COVID-999.png"
    model_path = "./model.pt"
    out_dir = "."

    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral']

    fmap_block = list()
    grad_block = list()

    img = cv2.imread(path_img)
    img_input = img_preprocess(img)

    net = torch.load(model_path)
    net.eval()

    net.layer4[-1].conv3.register_forward_hook(forward_hook)
    net.layer4[-1].conv3.register_backward_hook(backward_hook)

    img_input = img_input.to(device)
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("Predict: {}".format(classes[idx]))

    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    print(len(grad_block))

    grad_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    cam_show_img(img, fmap, grad_val, out_dir)
    
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_path = "./Covid/COVID/COVID-11.png"

net = torch.load("./model.pt")
finalconv_name = "layer4"
net.eval()
print(net)

features_blobs = []

def hook_feature(module, input, output): 
    print("hook input",input[0].shape)
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature) 

print(net._modules)

params = list(net.parameters()) 
weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) 


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []

    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w))) 

        cam = cam.reshape(h, w)

        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        cam_img = np.uint8(255 * cam_img)
        
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

transform = transforms.Compose({
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
})

img_pil = cv2.imread(img_path)
img = img_pil[:, :, ::-1]
img = np.ascontiguousarray(img)
img_tensor = transform(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
img_variable = img_variable.to(device)
logit = net(img_variable)


classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral']

h_x = F.softmax(logit, dim=1).data.squeeze() 

probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()


CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[2]])

print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])

img = cv2.imread(img_path)
img = cv2.resize(img,(256,256))
height, width, _ = img.shape

heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
cv2.imwrite('heatmap.jpg', heatmap)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
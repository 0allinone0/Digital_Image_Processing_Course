import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

def load_img(fn):
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(fn).convert('RGB')
    image = preprocessing(image) [None]
    return image.to(torch.float)


# Hyper Parameter
content_fn = 'lebron.jpg'
stype_fn = '고흐 style.jpg'
content_layers=['conv_4']
style_layers=['conv_1','conv_2', 'conv_3', 'conv_4', 'conv_5']
num_steps = 200
style_weight = 1e6
content_weight = 1

content_img = load_img(content_fn)
style_img = load_img(stype_fn)
input_img = content_img.clone() # torch.rand_like(content_img)
cnn = models.vgg19(pretrained=True).features.cuda().eval()



class ContentLoss(nn.Module):
    def __init__(self):
        super()._init_()

    def forward(self, input, target):
        target = target.detach()
        loss = nn.functional.mse_loss(input, target)
        return loss
    
class StyleLoss(nn.Module):
    def _init__(self):
        super()._init_()

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(c, h * w)
        G = torch.mm(features, features.t())
        return G.div(h * w)
    

    def forward(self, input, target_feature):
        G = self.gram_matrix(input)
        target = self.gram_matrix(target_feature).detach()
        loss = nn.functional.mse_loss(G, target)
        return loss
    
content_loss = ContentLoss()
style_loss = StyleLoss()


class Normalization(nn.Module):
    def _init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super()._init_()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).cuda ()
        self.std = torch.tensor(std).view(1, -1, 1, 1).cuda()

    def forward(self, img):
        return (img - self.mean)/ self.std

pre_processing = Normalization()

def feature_extractor(input, layers):
    f = pre_processing(input)
    outputs = []
    i = 0
    for layer in cnn.children():
        f = layer (f)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
            if name in layers:
                outputs.append(f)
        if len(outputs) == len(layers):
            break
    return outputs

style_features = feature_extractor(style_img, style_layers)
content_features = feature_extractor(content_img, content_layers)




#learning
optimizer = optim.Adam([input_img.requires_grad_()], lr=0.02)

print('Optimizing...')
step = 0
while step <= num_steps:
    optimizer.zero_grad()
    input_style = feature_extractor(input_img, style_layers)
    input_content = feature_extractor(input_img, content_layers)
    content_score = sum([content_loss(inp, tar) for inp, tar in zip(input_content, content_features)])
    style_score = sum([style_loss(inp, tar) for inp, tar in zip(input_style, style_features)])

    loss = style_weight * style_score / len(style_features) + content_weight * content_score / len(content_features)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")
    input_img.data.clamp_(0, 1)
    image = input_img.cpu().permute(0, 2, 3, 1).clone().squeeze(0).detach().numpy()
    cv2.imshow('img', image[..., ::-1])
    cv2.waitKey(1)
    step += 1

input_img.data.clamp_(0, 1)
image = input_img.cpu().permute(0, 2, 3, 1).clone().squeeze(0).detach().numpy()
cv2.imshow('img', image[..., ::-1])
cv2.waitKey(0)
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PrivacySuppressedExtractor(nn.Module):
    
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )
        self.psl_weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.projector(x)
        mask = torch.sigmoid(self.psl_weight)
        x = x * mask
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def load_image(path):
    img = Image.open(path).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)


def extract_features(model, image_path, device="cpu"):
    model.eval()
    t = load_image(image_path).to(device)
    with torch.no_grad():
        emb = model(t)
    return emb.squeeze(0).cpu().numpy()


def build_extractor(embedding_dim=256, device="cpu"):
    m = PrivacySuppressedExtractor(embedding_dim=embedding_dim)
    m.to(device)
    m.eval()
    return m
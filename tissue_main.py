import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, densenet121, efficientnet_b0, ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights
from transformers import ViTModel, SwinModel, ViTConfig, SwinConfig

# CNN Models
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet50(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = densenet121(weights=weights)
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
    
    def forward(self, x):
        return self.densenet(x)

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet-b0', pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.efficientnet = efficientnet_b0(weights=weights)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_classes
        )
    
    def forward(self, x):
        return self.efficientnet(x)

# Transformer Models
class ViTClassifier(nn.Module):
    def __init__(self, num_classes, model_name="google/vit-base-patch16-224", pretrained=True):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )
    
    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        # Get pooled output (ViT uses pooler_output or [CLS] token)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback: use [CLS] token (first token in sequence)
            pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

class SwinTransformerClassifier(nn.Module):
    def __init__(self, num_classes, model_name="microsoft/swin-base-patch4-window7-224", pretrained=True):
        super().__init__()
        if pretrained:
            self.swin = SwinModel.from_pretrained(model_name)
        else:
            config = SwinConfig.from_pretrained(model_name)
            self.swin = SwinModel(config)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.swin.config.hidden_size, num_classes)
        )
    
    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)

# Model Initialization (for direct execution only)
# This section only runs if tissue_main.py is executed directly
# When imported, models are created by train_comparative_study.py
if __name__ == '__main__':
    USE_PRETRAINED = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_dict = {
        'ResNet18': ResNet18Classifier(num_classes=8, pretrained=USE_PRETRAINED).to(device),
        'ResNet50': ResNet50Classifier(num_classes=8, pretrained=USE_PRETRAINED).to(device),
        'DenseNet121': DenseNet121Classifier(num_classes=8, pretrained=USE_PRETRAINED).to(device),
        'EfficientNet-B0': EfficientNetClassifier(num_classes=8, pretrained=USE_PRETRAINED).to(device),
        'ViT-B/16': ViTClassifier(num_classes=8, model_name="google/vit-base-patch16-224", pretrained=USE_PRETRAINED).to(device),
        'Swin-Tiny': SwinTransformerClassifier(num_classes=8, model_name="microsoft/swin-tiny-patch4-window7-224", pretrained=USE_PRETRAINED).to(device),
        'Swin-Base': SwinTransformerClassifier(num_classes=8, model_name="microsoft/swin-base-patch4-window7-224", pretrained=USE_PRETRAINED).to(device),
    }

    # Print model info
    for name, model in models_dict.items():
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {num_params/1e6:.1f}M parameters")
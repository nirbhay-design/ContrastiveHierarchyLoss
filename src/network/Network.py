import torch 
import torchvision 
import torch.nn as nn 

class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
        module_keys = list(model._modules.keys())
        self.feat_extractor = nn.Sequential()
        for key in module_keys[:-1]:
            module_key = model._modules.get(key, nn.Identity())
            if key != 'layer4':
                for params in module_key.parameters():
                    params.requires_grad = False
            self.feat_extractor.add_module(key, module_key)
        resnet_classifier = model._modules.get(module_keys[-1], nn.Identity())
        self.classifier = nn.Linear(in_features = resnet_classifier.in_features, out_features = num_classes)

    def forward(self, x):
        features = self.feat_extractor(x).flatten(1)
        scores = self.classifier(features.detach())
        return features, scores

if __name__ == "__main__":
    network = Network(num_classes=10)
    x = torch.rand(2,3,84,84)
    feat, score = network(x)
    print(network)
    print(feat.shape, score.shape)
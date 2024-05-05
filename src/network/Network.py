import torch 
import torchvision 
import torch.nn as nn 

class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = torchvision.models.resnet18(num_classes=num_classes)
        module_keys = list(model._modules.keys())
        self.feat_extractor = nn.Sequential()
        for key in module_keys[:-1]:
            self.feat_extractor.add_module(key, model._modules.get(key, nn.Identity()))
        self.classifier = model._modules.get(module_keys[-1], nn.Identity())

    def forward(self, x):
        features = self.feat_extractor(x).flatten(1)
        scores = self.classifier(features)
        return features, scores

if __name__ == "__main__":
    network = Network(num_classes=10)
    x = torch.rand(2,3,224,224)
    feat, score = network(x)
    print(feat.shape, score.shape)
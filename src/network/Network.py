import torch 
import torchvision 
import torch.nn as nn 

class Network(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        module_keys = list(model._modules.keys())
        self.feat_extractor = nn.Sequential()
        for key in module_keys[:-1]:
            module_key = model._modules.get(key, nn.Identity())
            self.feat_extractor.add_module(key, module_key)
        resnet_classifier = model._modules.get(module_keys[-1], nn.Identity())

        resnet_cls_infeature = resnet_classifier.in_features
        self.classifier = nn.Linear(in_features = resnet_cls_infeature, out_features = num_classes)

    def forward(self, x):
        features = self.feat_extractor(x).flatten(1)
        scores = self.classifier(features)
        return features, scores

if __name__ == "__main__":
    network = Network(num_classes=10)
    x = torch.rand(2,3,84,84)
    feat, score = network(x)
    print(network)
    print(feat.shape, score.shape)
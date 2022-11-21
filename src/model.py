import torch
import torch.nn as nn
import architecture.resnet3d as resnet3d
import architecture.mobilenet as mobilenet
import architecture.resnext as resnext
from efficientnet_pytorch_3d import EfficientNet3D


class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes=2, is_pretrained=True, dropout_rate=0.5):
        super().__init__()
    
        if model_name == "resnet10":
            self.base_model = resnet3d.resnet10_3d(input_channels = 1, num_classes=num_classes)
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) 
        
        elif model_name == "resnet18":
            self.base_model = resnet3d.resnet18_3d(input_channels = 1, num_classes=num_classes)
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) 
        
        elif model_name == "resnet34":
            self.base_model = resnet3d.resnet34_3d(input_channels = 1, num_classes=num_classes)
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) 

        elif model_name == "resnet50":
            self.base_model = resnet3d.resnet50_3d(input_channels = 1, num_classes=num_classes)
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) 

        elif model_name == "efficientnet":
            self.base_model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)
            classifier_input_size = self.base_model._fc.in_features
            self.net._fc = nn.Linear(classifier_input_size, out_features=num_classes, bias=True)

        elif model_name == "mobilenet":
            self.base_model = mobilenet.MobileNet(input_channel=1, num_classes=2)

        elif model_name == "resnext50":
            self.base_model = rexnet.resnext50(input_channels = 1, num_classes=num_classes)
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) 
    def foward(self,img):
        return self.base_model(img)
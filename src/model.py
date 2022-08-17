import torch
import torch.nn as nn
import resnet3d


class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes=2, is_pretrained=True, dropout_rate=0.5):
        super().__init__()
    
        if model_name == "resnet10":
        # Load Resnet50 with pretrained ImageNet weights
            self.base_model = resnet_3d.resnet10_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
        
        elif model_name == "resnet18":
        # Load Resnet50 with pretrained ImageNet weights
            self.base_model = resnet_3d.resnet18_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
        
        elif model_name == "resnet34":
            self.base_model = resnet_3d.resnet34_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())

        elif model_name == "resnet50":
            self.base_model = resnet_3d.resnet50_3d(pretrained = is_pretrained, input_channels = 1, num_classes=num_classes)
            # replace the last layer with a new layer that have `num_classes` nodes, followed by Sigmoid function
            
            classifier_input_size = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(classifier_input_size, num_classes, bias=True)) ### bias = True ????
            #                 nn.LogSoftMax())
    
    def foward(self,img):
        return self.base_model(img)
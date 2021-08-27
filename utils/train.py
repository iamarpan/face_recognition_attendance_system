import torch
import yaml
import os
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
from utils.models.inception_resnet_v1 import InceptionResnetV1
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
config_file = './config.yaml'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

    
class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x


def read_yaml():
    with open(config_file,"r") as f:
        return yaml.safe_load(f)


class Train(nn.Module):

    def __init__(self):
        super(Train,self).__init__()
        self.transform = transform.Compose([
                                transform.RandomHorizontalFlip(),
                                transform.ToTensor(),
                                transform.Scale((224,224)),
                                transform.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                transform.RandomRotation(5, resample=False,expand=False, center=None),
                                transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.config = read_yaml()
        self.batch_size = self.config['NETWORK']['BATCH_SIZE']
        self.epoch = self.config['NETWORK']['EPOCH']
        self.folderPath = self.config['DATASET']['FOLDER_PATH']
        self.classes = next(os.walk(self.folderPath))[1]
        self.model_path = self.config['MODEL']['PATH']
        self.train_set = None
        self.network = None


    def performETL(self,batch_size):
        self.train_set = torchvision.datasets.ImageFolder(root=self.folderPath,transform=self.transform)        
        dataloader = torch.utils.data.DataLoader(self.train_set,batch_size=batch_size,shuffle=True)
        return dataloader


    def modifyNetwork(self,classes):
        resnet_model = InceptionResnetV1(pretrained='vggface2',classify=False,num_classes=len(classes))
        list(resnet_model.children())[-6:]
        layer_list = list(resnet_model.children())[-5:]
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-5])
        for param in resnet_model.parameters():
            param.requires_grad = False
        resnet_model.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        resnet_model.last_linear = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=1792,out_features=512,bias=False),
                normalize()
                )
        resnet_model.logits = nn.Linear(layer_list[2].out_features,len(classes))
        resnet_model.softmax = nn.Softmax(dim=1)
        resnet_model = resnet_model.to(device)
        self.network = resnet_model
        return self.network

    def generateModel(self,model):
        dataloader = torch.utils.data.DataLoader(self.train_set,batch_size=1)
        onnx_model_path = self.model_path
        images,labels = next(iter(dataloader))
        torch.onnx.export(self.network,images.to(device).float(),onnx_model_path,verbose=True)
        return 

    def train(self):
        self.network = self.modifyNetwork(self.classes)
        dataloader = self.performETL(self.batch_size)
        optimizer = optim.Adam(self.network.parameters(),lr=0.0001)
        for _ in tqdm(range(self.epoch)):
            for image,label in dataloader:
                images,labels = image.to(device),label.to(device)
                optimizer.zero_grad()
                output = self.network(images)
                loss = F.cross_entropy(output,labels)
                loss.backward()
                optimizer.step()
                
                del images,labels,output 
                torch.cuda.empty_cache()
        return self.generateModel(self.network)




    



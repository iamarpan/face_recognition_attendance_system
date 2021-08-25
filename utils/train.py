import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn
from models.inception_resnet_v1 import InceptionResnetV1
import torch.optim as optim
import torch.nn.functional as F

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


class Train(nn.Module):

    def __init__(self,folderPath,transform,classes,config):
        super(Train,self).__init__(self)
        self.folderPath = folderPath
        self.transform = transform
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.classes = classes
        self.train_set = None
        self.network = None


    def performETL(self,batch_size):
        self.train_set = torchvision.datasets.ImageFolder(root=self.folderPath,transform=self.transform)        
        dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
        return dataloader


    def modifyNetwork(self,classes):
        resnet_model = InceptionResnetV1(pretrained='vggface2',classify=False,num_classes=len(classes))
        layer_list = list(resnet_model.children())[-5:0]
        resnet_model = nn.Sequential(*list(resnet_model.children())[:-5])
        for param in resnet_model.parameters():
            param.requires_grad = False
        resnet_model.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
        resnet_model.last_linear = nn.Sequential(
                Flatten(),
                nn.Linear(in_features=1792,out_features=512,bias=False),
                normalize()
                )
        resnet_model.logits = nn.Linear(layer_list[3].in_features,len(classes))
        resnet_model.softmax = nn.Softmax(dim=1)
        resnet_model = resnet_model.to(device)
        self.network = resnet_model
        return self.network

    def generateModel(self,model):
        dataloader = torch.utils.data.DataLoader(self.train_set,batch_size=1)
        onnx_model_path = './model_file/model.onnx'
        images,labels = next(iter(dataloader))
        torch.onnx.export(self.network,images.to(device).float(),onnx_model_path,verbose=True)
        return 

    def train(self):
        self.network = self.modifyNetwork(self.classes)
        dataloader = self.performETL(self.batch_size)
        optimizer = optim.Adam(network.parameters(),lr=0.0001)
        
        for _ in range(epoch):
            for image,label in dataloader:
                images,labels = image.to(device),label.to(device)
                optimizer.zero_grad()
                output = network(images)
                loss = F.cross_entropy(output,labels)
                loss.backward()
                optimizer.step()
                
                del images,labels,output 
                torch.cuda.empty_cache()
        return self.generateModel(self.network)


if __name__ == '__main__':
    train = Train()
    train.train()


    



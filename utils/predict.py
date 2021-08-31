import onnxruntime as rt
import numpy as n
import torch.nn.functional as F
import torch
import torchvision.transforms as transform
import cv2
import yaml
import json
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import ast
config_file = './config.yaml'


def read_txt(file_path):
    with open(file_path,'r') as f:
        classes = f.read()
    return classes


def read_yaml():
    with open(config_file,"r") as f:
        return yaml.safe_load(f)


class Prediction:
    def __init__(self):
        self.mtcnn = MTCNN()
        self.config = read_yaml() 
        self.transform = transform.Compose([
                                transform.RandomHorizontalFlip(),
                                transform.ToTensor(),
                                transform.Scale((224,224)),
                                transform.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                transform.RandomRotation(5, resample=False,expand=False, center=None),
                                transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])


    def _draw(self,bbox,text,frame):
        print(bbox)
        start_point = (int(bbox[0]),int(bbox[1]))
        end_point = (int(bbox[2]),int(bbox[3]))
        color = (255,0,0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        frame = cv2.rectangle(frame,start_point,end_point,color,thickness)
        frame = cv2.putText(frame,text,start_point,font,fontScale,color,thickness,cv2.LINE_AA)
        return frame


    def _inference(self,img):
        session = rt.InferenceSession(self.config['MODEL']['PATH'])
        classes = ast.literal_eval(read_txt(self.config['MODEL']['CLASSES']))
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        img = img.reshape((1,3,224,224))
        data = json.dumps({'data':img.tolist()})
        data = np.array(json.loads(data)['data']).astype('float32')
        result = session.run([output_name],{input_name:data})
        prediction = torch.argmax(F.softmax(torch.from_numpy(np.array(result[0].squeeze())),dim=0),dim=0)
        prediction = prediction.item()
        for key,value in classes.items():
            if value == prediction:
                prediction = key
                break
        result = prediction.split("_")[-1]
        return result


    def predictUser(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            try:
                bbox,prob,ld = self.mtcnn.detect(frame,landmarks=True)
            except:
                continue
            if bbox is not None:
                newbbox = bbox[0]
                width = int(newbbox[2]-newbbox[0])
                height = int(newbbox[3]-newbbox[1])
                roi = frame[int(newbbox[1]):int(newbbox[1])+height,int(newbbox[0]):width+int(newbbox[0])]
                roi = Image.fromarray(np.uint8(roi)).convert('RGB')
                roi = self.transform(roi)
                inference = self._inference(roi)
                print(prob)
                text = str(inference) + "  " + str(round(prob[0],4))
                frame = self._draw(bbox[0],text,frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()





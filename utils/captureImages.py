import cv2
import os
from facenet_pytorch import MTCNN


class CaptureImages:

    def __init__(self,args):
        self.mtcnn = MTCNN()
        self.max_images = args['faces']
        self.file_path = args['output']

    def _draw(self,bbox,prob,ld,frame):
        start_point = (int(bbox[0]),int(bbox[1]))
        end_point = (int(bbox[2]),int(bbox[3]))
        color = (255,0,0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        frame = cv2.rectangle(frame,start_point,end_point,color,thickness)
        frame = cv2.putText(frame,str(prob),start_point,font,fontScale,color,thickness,cv2.LINE_AA)
        return frame


    def capture(self):
        cap = cv2.VideoCapture(0)
        count = 0
        os.makedirs(self.file_path,exist_ok=True)
        while True:
            ret,frame = cap.read()
            bbox,prob,ld = self.mtcnn.detect(frame,landmarks=True)
            if bbox is not None:
                frame = self._draw(bbox[0],prob,ld[0],frame)
            cv2.imshow('frame',frame)
            if prob[0] and prob[0]>0.9:
                try:
                    newbbox = bbox[0]
                    width = int(newbbox[2]-newbbox[0])
                    height = int(newbbox[3] - newbbox[1])
                    roi = frame[int(newbbox[1]):int(newbbox[1])+height,int(newbbox[0]):width+int(newbbox[0])]
                except Exception as e:
                    continue
                count+=1
                cv2.imwrite(self.file_path + '/'+str(count)+'.jpg',roi)
                if cv2.waitKey(1) & count==self.max_images:
                    break
        cap.release()
        cv2.destroyAllWindows()


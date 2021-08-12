import cv2
from facenet_pytorch import MTCNN


class CaptureImages:

    def __init__(self,file_path,max_images=20):
        self.mtcnn = mtcnn
        self.max_images = max_images
        self.file_path = 'datasets'

    def _draw(self,bbox,prob,ld,frame):
        start_point = (bbox[0],bbox[1])
        end_point = (bbox[2],bbox[3])
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
        while True:
            ret,frame = cap.read()

            bbox,prob,ld = self.mtcnn.detect(frame,landmarks=True)
            if prob[0]:
                frame = self._draw(bbox[0],prob,ld[0],frame)
                try:
                    newbbox = bbox[0]
                    width = int(newbbox[2]-newbbox[0])
                    height = int(newbbox[3] - newbbox[1])
                    roi = frame[int(newbbox[1]):int(newbbox[1])+height,int(newbbox[0]):width+int(newbbox[0])]
                    cv2.imshow('frame2',roi)
                except Exception as e:
                    print("coming in except",e)
                    continue
                count+=1
                cv2.imwrite(self.file_path + '/'+str(count)+'.jpg',roi)
                print(count,self.max_images)
            if cv2.waitKey(1) & 0xff==ord('q'):
                break
            elif count==self.max_images:
                break
        cap.release()
        cv2.destroyAllWindows()

mtcnn = MTCNN()
images = CaptureImages(mtcnn)
images.capture()


#install packages
pip install opencv-python
pip install matplotlib

import cv2
import matplotlib.pyplot as plt

#config setup
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

#model config
model = cv2.dnn_DetectionModel(frozen_model,config_file)

#defining class labels
classLabels = []
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

#verifying weight class    
print(classLabels)
print(len(classLabels))

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

#read image
img = cv2.imread('test.jpeg') #check image name and format proerly

#image in BGR format (Blue, green, red)
plt.imshow(img)

#image conversion to RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ClassIndex, confidece, bbox = model.detect(img,confThreshold=0.5)
print(ClassIndex) #index of object detected in image

font_scale=3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255, 0, 0), 2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0, 255, 0), thickness=3)
    
#for detection in video 
cap = cv2.VideoCapture("P1020613.mp4") #check image name and format proerly. To open webcam, just pass 1 as parameter inside VideoCapture function

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IDError("Cannot open video")
    
font_scale=3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    
    ClassIndex, confidece, bbox = model.detect(frame,confThreshold=0.55)
    
    print(ClassIndex)
    
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255, 0, 0), 2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0, 255, 0), thickness=3)
                
            cv2.imshow('Object detection test',frame)
            
            if cv2.waitKey(2) & 0xff == ord('q'):
                break
            
            cap.release()
            cv2.destroyAllWindows()

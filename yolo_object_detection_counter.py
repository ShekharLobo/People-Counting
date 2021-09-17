import cv2
import numpy as np
from centroid import CentroidTracker
from trackableobject import TrackableObject
import dlib

def show(img):
    ret, img=cv2.imencode('.jpg', img)
    img= img.tobytes()
    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
    print("test")

# Load Yolo
#net = cv2.dnn.readNet("yolov3_EmrataGal.cfg", "yolov3_EmrataGal_last.weights")
net = cv2.dnn.readNet("weights\\yolov3\\yolov3.cfg", "weights\\yolov3\\yolov3.weights")
#net = cv2.dnn.readNet("weights\\yolov3_custom\\yolov3_custom.cfg", "weights\\yolov3_custom\\yolov3_custom_1000.weights")
classes = []
with open("weights\\yolov3\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
out=None
# Loading image
#img = cv2.imread("D:\\Wallpaper\\tom-cruise-as-ethan-hunt-in-mission-impossible-fallout-2018-l2-3840x2400.jpg")
#img = cv2.imread("D:\\Train_Images\\Celebrities\\Kendall Jenner\\3c12411-01.jpeg")
#img = cv2.resize(img, None, fx=0.8, fy=0.8)
#height, width, channels = img.shape
#Loading videos
#cap = cv2.VideoCapture('D:\\Train_Images\\Road traffic video for object recognition.mp4')
cap = cv2.VideoCapture('C:\\Users\\Shekhar\\Downloads\\example_01.mp4')
#height=int(cap.get(4))
#width=int(cap.get(3))

count =0
ct = CentroidTracker(maxDisappeared=40)
totalUp=totalDown=0
trackers=[]
x=y=[]
trackableObjects = {}
list=list1=empty=empty1=[]

while(True):
    ret, img = cap.read()
    #img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channel = img.shape
    #print(height, width)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_name)

    boxes =[]
    confidences= trackers = []
    class_ids = []
    rects=[]
    listup=[]
    listdown=[]
    direction=0
    #count=0
    cv2.line(img, (0,height//2), (width,height//2), (0,255,255), 2)

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.25:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

                
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size =(len(boxes),3))
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            if (label=="car" or label=="truck" or label=="person"):
                rects.append([x,y,x+w,y+h])
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                #cv2.putText(img,objectID, (centroid[0],centroid[1]),font,2,color,2)
                #cv2.putText(img,label + " " + confidence, (x,y+40),font,2,color,2)
                #rects = dlib.rectangle(x,y,x+w,y+h)
    objects=ct.update(rects)


    for(objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)
        #print(to)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            #print()
            y = [c[1] for c in to.centroids]
            #if(centroid[1]>350):
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            #else:
            #direction=1
            #print(y, np.mean(y), centroid[1], direction)

            #print(to.objectID, to.centroids[0][1])

            #check direction
            #flag1=0
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < height//2:
                    totalUp += 1
                    empty.append(totalUp)
                    to.counted = True
                    print(str(to.objectID)+ " Is gping up")

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] >= height//2:
                    totalDown += 1
                    empty1.append(totalDown)
                    to.counted = True
                            
                x = []
                # compute the sum of total people inside
                x.append(len(empty1)-len(empty))
                #print("Total people inside:", x)
                
            #print(totalDown)     
             
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
   
        #if list1!=[]:
            #for j in list1:
               # print (j[0])
            #for(j1,j2) in list1:
                #if (j[0] == objectID):
                    #flag1=1
                    #if(j[1]-centroid[1]<0):
                        #print(j[0]+" moving down")
                    #elif(j[1]-centroid[1]==0):
                        #print(j[0]+" standby")
                    #elif(j[1]-centroid[i]>0):
                        #print(j[0]+" moving up")

        #if(flag1 != 1):
            #list1.append([objectID,centroid[1]])
        #else:
        #print(list1)


        flag=0
        cv2.putText(img,str(objectID), (centroid[0],centroid[1]),font,2,color,2)
        #if (centroid[1]<=502):
            #print(objectID)
            #for i in list:
                #if (i==objectID):
                    #list.remove(objectID)
        #if (centroid[1]>502):
            #j[0]=objectID
            #if list==[]:
                #list.append(objectID)
                #count+=1


            #for i in list:
                #if(i == objectID):
                    #flag=1
                    #break
            #if flag!=1:
                #list.append(objectID)
                #count+=1
            #else:
                #flag=0

                    #print(i)
                        #list.append(objectID)
            #print(list)

    #print(totalDown, totalUp)
    cv2.putText(img, "Enter: "+str(totalDown), (5,40),font,2,(0,255,255),2)
    cv2.putText(img, "Exit: "+str(totalUp), (205,40),font,2,(0,255,255),2)
    #show(img)   
    #cv2.imshow('img',img)
    #if out is None:
        #out = cv2.VideoWriter('output_people.avi',cv2.VideoWriter_fourcc(*"MJPG"), 15, (img.shape[1], img.shape[0]), True)
    #if out is not None:
        #out.write(img)
            
        

    if cv2.waitKey(1) == ord('q'):
        break
    
#cap.release()
#out.release()
cv2.destroyAllWindows()
# Smart-Road-Safety-System

## BOM

Here i build the BOM of all the components. This are the thing to build the whole project 

https://docs.google.com/spreadsheets/d/17JcF96xBJZzzzK9_Rad6b0UQn5YGpifdfMuCeLcltTo/edit?gid=0#gid=0

<img width="3420" height="2052" alt="image" src="https://github.com/user-attachments/assets/cba1f328-89ec-408c-b15b-55133caa70a3" />
<img width="3420" height="2066" alt="image" src="https://github.com/user-attachments/assets/dde37857-433a-449f-b890-413c6f9c476f" />
<img width="3416" height="1912" alt="image" src="https://github.com/user-attachments/assets/aacb94f9-4cec-4bf4-87a4-98535efe90c6" />
<img width="3416" height="1220" alt="image" src="https://github.com/user-attachments/assets/d125c0be-01a1-4d6c-a13d-458c53e06be0" />
<img width="3420" height="1902" alt="image" src="https://github.com/user-attachments/assets/4fe7b67e-a4b4-42d5-9e67-328bba825536" />
<img width="3420" height="2054" alt="image" src="https://github.com/user-attachments/assets/1fb71e87-2956-4553-a013-478679f9a962" />







## Coding - ML

Here i did some couple of thing that how it will detect if the driver is sleeping or not and if user also talk to someone then it will warn him to avoid accident

main.py

```python
from subprocess import call
import multiprocessing

def drowsiness():
    call(["python", "drowsiness.py"])

def talking():
    call(["python", "phone.py"])

drowsy = multiprocessing.Process(target=drowsiness)
talk = multiprocessing.Process(target=talking)

if __name__ == '__main__':
    talk.start()
    drowsy.start()
```


phone.py

```python
import numpy as np
import cv2
from scipy.spatial import distance as dist
import os
from playsound import playsound
os.system("cls")

talking_in_phone = False


Known_distance = 30 
Known_width = 5.7 
thres = 0.5  
nms_threshold = 0.2  

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

font = cv2.FONT_HERSHEY_PLAIN
fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX

cap = cv2.VideoCapture(0)  
face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
Distance_level = 0
classNames = []
with open("coco.names", "r") as f:
    classNames = f.read().splitlines()
print(classNames)
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"


face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length



def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance



def face_data(image, CallOut, Distance_level):
    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for x, y, h, w in faces:
        line_thickness = 2
       
        LLV = int(h * 0.12)
   

    
        cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
        cv2.line(
            image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness
        )
        cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)

        face_width = w
        face_center = []

        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
        if Distance_level < 10:
            Distance_level = 10

        if CallOut == True:
       
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (ORANGE), 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (YELLOW), 20)
            cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), (GREEN), 18)

    return face_width, faces, face_center_x, face_center_y


ref_image = cv2.imread("lena.png")

ref_image_face_width, _, _, _ = face_data(ref_image, False, Distance_level)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)
print(Focal_length_found)

while True:
    _, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    face_width_in_frame, Faces, FC_X, FC_Y = face_data(frame, True, Distance_level)

    if len(classIds) != 0:
        for i in indices:
            i = i
            box = bbox[i]
            confidence = str(round(confs[i], 2))
            color = Colors[classIds[i] - 1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(
                frame,
                classNames[classIds[i] - 1] + " " + confidence,
                (x + 10, y + 20),
                font,
                1,
                color,
                2,
            )
    for face_x, face_y, face_w, face_h in Faces:
        if face_width_in_frame != 0:
            Distance = Distance_finder(
                Focal_length_found, Known_width, face_width_in_frame
            )
            Distance = round(Distance, 2)
           
            Distance_level = int(Distance)

            cv2.putText(
                frame,
                f"Distance {Distance} Inches",
                (face_x - 6, face_y - 6),
                fonts,
                0.5,
                (BLACK),
                2,
            )

    if cv2.waitKey(1) == ord("q"):
        break

    status, photo = cap.read()
    l = len(bbox)
    frame = cv2.putText(
        frame,
        str(len(bbox)) + " Object",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    stack_x = []
    stack_y = []
    stack_x_print = []
    stack_y_print = []
    global D

    if len(bbox) == 0:
        pass
    else:
        for i in range(0, len(bbox)):
            x1 = bbox[i][0]
            y1 = bbox[i][1]
            x2 = bbox[i][0] + bbox[i][2]
            y2 = bbox[i][1] + bbox[i][3]

            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            stack_x.append(mid_x)
            stack_y.append(mid_y)
            stack_x_print.append(mid_x)
            stack_y_print.append(mid_y)

            frame = cv2.circle(frame, (mid_x, mid_y), 3, [0, 0, 255], -1)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)

        if len(bbox) == 2:
            D = int(
                dist.euclidean(
                    (stack_x.pop(), stack_y.pop()), (stack_x.pop(), stack_y.pop())
                )
            )
            frame = cv2.line(
                frame,
                (stack_x_print.pop(), stack_y_print.pop()),
                (stack_x_print.pop(), stack_y_print.pop()),
                [0, 0, 255],
                2,
            )
        else:
            D = 0

        if D < 250 and D != 0:
            frame = cv2.putText(
                frame,
                "!!MOVE AWAY!!",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                [0, 0, 255],
                4,
                playsound("drive.mp3")
            )
            talking_in_phone = True;

        frame = cv2.putText(
            frame,
            str(D / 10) + " cm",
            (300, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Output", frame)
        if cv2.waitKey(100) == 13:
            break
print(talking_in_phone)
cap.release()

cv2.destroyAllWindows()

```

drowsiness.py
```python
import cv2
import numpy as np
import dlib
from imutils import face_utils
import serial
from playsound import playsound


arduino = serial.Serial(port='/dev/cu.usbmodem21401', baudrate=9600, timeout=1)

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
relay_on = False  
debug = True 


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2 
    elif ratio > 0.21 and ratio <= 0.25:
        return 1 
    else:
        return 0  

# Main loop
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0: 
        sleep = 0
        drowsy = 0
        active = 0
        status = "No Face Detected"
        color = (255, 255, 255) 
        if not relay_on:
                    arduino.write(b'SLEEP\n')
                    if debug:
                        print("Sleeping detected. Relay turned ON.")
                    relay_on = True

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(
            landmarks[36],
            landmarks[37],
            landmarks[38],
            landmarks[41],
            landmarks[40],
            landmarks[39],
        )
        right_blink = blinked(
            landmarks[42],
            landmarks[43],
            landmarks[44],
            landmarks[47],
            landmarks[46],
            landmarks[45],
        )

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                playsound("weak.mp3")
                if not relay_on:
                    arduino.write(b'SLEEP\n')
                    if debug:
                        print("Sleeping detected. Relay turned ON.")
                    relay_on = True

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                playsound("drowsy.mp3")
                if not relay_on:
                    arduino.write(b'DROWSY\n')
                    if debug:
                        print("Drowsiness detected. Relay turned ON.")
                    relay_on = True

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                if relay_on:
                    arduino.write(b'ACTIVE\n')
                    if debug:
                        print("Active state detected. Relay turned OFF.")
                    relay_on = False

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)


    if len(faces) == 0:
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    
    cv2.imshow("Car Driving", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
arduino.close()
```





<img width="3420" height="2224" alt="image" src="https://github.com/user-attachments/assets/68fdaef1-c802-4dc4-bbe4-2812ed55049c" />




## Components and circuit

Here i selected the required devices and understanding how they connect together. The aim of this project is to improve road safety by detecting vehicles, obstacles, and unsafe conditions, and then providing alerts when necessary.

I identified main control units like Arduino and Raspberry Pi, along with sensors that can detect distance, movement, and alcohol presence. GPS and GSM are used for location tracking and sending emergency messages. Displays and buzzers help show warnings, while cameras and RFID support monitoring and identification. Motors, relays, and power units are included to control actions and supply energy.

<img width="1118" height="1383" alt="image" src="https://github.com/user-attachments/assets/5df291d3-d299-44ff-b536-1d5e146d2fd8" />



Here is the circuit diagram




<img width="1126" height="1556" alt="image" src="https://github.com/user-attachments/assets/151c4ef4-b15d-4a70-93e6-876a75a35251" />






## Prototyping 

To testing this thing i will build a prototype. Before that leets plan where and how all the components will set to the vehicle. 

What it should look in the real car.

<img width="3072" height="1343" alt="image" src="https://github.com/user-attachments/assets/79a2ae0b-9e29-4788-bfbc-08c2f313a5f2" />



How my prototype will be 

<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/66c260a4-8909-43d9-b2c2-948709ccc7b3" />




## Work flow 

### **How it will works**

As there is a lots of reasons of happening accident i tried to solve this issue. 

Task view

- Start the vehicle after scanning the driver
- Check driver condition (Awake and not drunk)
- Over speeding ( According to the place speed limit will be set )
- In the night the High bim light will convert to low bim light automatically 
- Send alert on vehicle accident

<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/230e7333-01df-4dff-b0ab-0b37ada2c107" />


Here is the flow chat show the thing will goes.

<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/5470b76f-4229-40c4-8c7b-12c6f9303a96" />



## Planning of the workflow

### Intro 

Now a days, we often heard about road accident, it may happen between two, three even four wheeler vehicle. there is couple of reasons of being happen. I have planned to make a device which i can install to these device and to alert the driver and prevent the accident.



Leets breakdown the thoughts then i will go to make the work flow 



<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/520af798-3490-4189-8fdb-0818ec35d4d4" />







import cv2
import numpy as np
from ultralytics import YOLO

camera = cv2.VideoCapture(0)
model = YOLO('yolov8n-face.pt')
#looping through each frame 
while True:
    success, frame = camera.read()
    if success is not True:
        break
    else:
        #frame ko model ke input me deni h 
        predictions = model.predict(frame, show=False)
        boxes = predictions[0].boxes
        boxdata = []
        for box in boxes:
            boxdata.append(box.data.tolist()[0])

        bx = np.array(boxdata)
        
        for cord_face in bx:
            x1, y1, x2, y2, acc, _  = cord_face
            acc = round(acc, 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            pt1 , pt2 = (x1, y1), (x2, y2)
            cv2.rectangle(frame, pt1, pt2, (0,0,255), 1)
            cv2.putText(frame, str(acc), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,255,0), 1 )
        cv2.imshow("frame",frame)

    if  cv2.waitKey(10) & 0xff == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

print("hello piyush and isha")
    
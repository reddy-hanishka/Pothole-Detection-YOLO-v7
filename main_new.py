# importing required libraries

import torch
import cv2
import time
import matplotlib.pyplot as plot_boxes
from datetime import datetime
import os
import geocoder
from playsound import playsound

c=0
# Main fucntion

g= geocoder.ip('me')

from twilio.rest import Client

SID = 'AC8a580ada9d0ee248800603c8fae145dd'
AUTH_TOKEN = '42f589f45939a932bc11595d524a9b95'
cl = Client(SID, AUTH_TOKEN)



def main(image_path = None, video_path = None, video_out = None,webcam=None):

    print("Loading YOLOv7 model . . . ")
    ## loading our custom yolov5 trained model
    model =  torch.hub.load("C:/Users/salla/OneDrive/Desktop/major data/zip/yolov7", 'custom', source ='local', path_or_model='C:/Users/salla/OneDrive/Desktop/major data/zip/yolov7/pothole/yolov7_pothole/weights/best.pt', force_reload=True) ### The model is stored locally
    class_names = model.names ### class names in string format
    print(class_names)

    if image_path:
        print("Image Path:", image_path)
        frame = cv2.imread(image_path)
        width = frame.shape[1]
        print("Width:", width)
        height = frame.shape[0]
        print("Height:", height)
        print("Dimension:", frame.shape)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = model_pred(frame, model = model)

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = class_names,width=width,height=height)

        # cv2.namedWindow("Image Window", cv2.WINDOW_NORMAL) 

        while True:
            # frame = cv2.cvtCsolor(frame,cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Done")
            # cv2.imwrite("s46.jpg",frame)
            print("Exiting from the window . . .")
            cv2.imwrite("final_output37.jpg",frame) # This line allows us to save our predicted image as output file.
            break


    elif video_path or webcam:
        print("Working with Video File . . .")

        ## reading the video
        if webcam:
            cap = cv2.VideoCapture(0)
        elif video_path:
            cap=cv2.VideoCapture(video_path)
            frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
            print("Total Frames=",frames)
            fps_count = int(cap.get(cv2.CAP_PROP_FPS))

        if video_out: # creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##('XVID')
            out = cv2.VideoWriter(video_out, codec, fps, (width, height))

        # print(width, height)
        print("FPS:",fps)

        

        # assert cap.isOpened()
        frame_no = 1

        m=0
        interval=20
        

        # cv2.namedWindow("Video_Capture", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()

            cv2.putText(frame,"Latitude and Longitude"+f"{g.latlng}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
                        
            if ret:
                print("Working on frame by frame video file . . .")
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = model_pred(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                if webcam:
                    frame = plot_boxes(results, frame,class_names,width, height)
                elif video_path:
                    frame = plot_boxes(results, frame,class_names,width, height,fps=fps_count,frame_no=frame_no)
                cv2.imshow("Captured Video", frame)
                if video_out:
                    print("Saving out predicted output video . . .")
                    out.write(frame)
                # if webcam:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break  
            frame_no += 1
            

        #print("Present Frame:",frame_no)
        print('Exiting from all the windows . . .')    
        cap.release()
        # closing all available windows
        cv2.destroyAllWindows()


def model_pred (frame, model):
    #print(frame)
    print("Sit tight, Your work is in progress...")
    results = model(frame)
    # results.show() # This will display your output
    print(results.xyxyn)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    print(labels, cordinates)

    return labels, cordinates


def plot_boxes(results, frame,classes,width,height,fps=None,frame_no=None,l=[], interval=int(50)):
    labels, img_cords = results
    #print(labels, img_cords)
    n_detections = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    #print("Total Number of Detections made:", n_detections)
    #print("Looping Through the Detections . . .")
    l.append(n_detections)
    area_img=width*height

    if len(l)%interval == 0:
        cv2.putText(frame,"Messege Sent with Location...", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
        sms = "Latitude and Longitude : "+f"{g.latlng}"
        cl.messages.create(body=sms , from_='+15855171523', to='+918498823361')

    #looping through the detections
    c=0
    for i in range(n_detections):
        bbox_cords = img_cords[i]
        #print(bbox_cords)
        if int(bbox_cords[2]*x_shape) < width/2: # threshold value for detection. bbox_cords[4] is the confidence of prediction and we discard every detection with confidence less than 0.7.
            if float(bbox_cords[4]) > 0.6:
                x1, y1, x2, y2 = int(bbox_cords[0]*x_shape), int(bbox_cords[1]*y_shape), int(bbox_cords[2]*x_shape), int(bbox_cords[3]*y_shape) ## BBOx coordniates
                label_name = classes[int(labels[i])]            
                #print(x1, y1, x2, y2)
                w_frame=x2-x1
                h_frame=y2-y1
                label_name = 'pothole Left'
                #print(label_name)
                area=round(float(w_frame*h_frame*100/area_img),2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## Drawing Bouding Boxes
                cv2.putText(frame, label_name + f" {round(float(bbox_cords[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                cv2.putText(frame, "Width: "+f"{w_frame}"+ " Height: "+f"{h_frame}", (x1, y1-int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                cv2.putText(frame,"Potholes "+f"{n_detections}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
                playsound(r'C:/Users/salla/OneDrive/Desktop/major data/zip/left.mp3')

        elif int(bbox_cords[2]*x_shape) > width/2: # threshold value for detection. bbox_cords[4] is the confidence of prediction and we discard every detection with confidence less than 0.7.
            if float(bbox_cords[4]) > 0.6:
                x1, y1, x2, y2 = int(bbox_cords[0]*x_shape), int(bbox_cords[1]*y_shape), int(bbox_cords[2]*x_shape), int(bbox_cords[3]*y_shape) ## BBOx coordniates
                label_name = classes[int(labels[i])]            
                #print(x1, y1, x2, y2)
                w_frame=x2-x1
                h_frame=y2-y1
                label_name = 'pothole Right'
                #print(label_name)
                area=round(float(w_frame*h_frame*100/area_img),2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## Drawing Bouding Boxes                
                cv2.putText(frame, label_name + f" {round(float(bbox_cords[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                cv2.putText(frame, "Width: "+f"{w_frame}"+ " Height: "+f"{h_frame}", (x1, y1-int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                cv2.putText(frame,"Potholes "+f"{n_detections}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
                playsound(r'C:/Users/salla/OneDrive/Desktop/major data/zip/right.mp3')
            
        else:
            if float(bbox_cords[4]) > 0.6:
                label_name = 'pothole Centre'            
                #print(x1, y1, x2, y2)
                w_frame=x2-x1
                h_frame=y2-y1
                #print(label_name)
                area=round(float(w_frame*h_frame*100/area_img),2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## Drawing Bouding Boxes
                cv2.putText(frame, label_name + f" {round(float(bbox_cords[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                cv2.putText(frame, "Width: "+f"{w_frame}"+ " Height: "+f"{h_frame}", (x1, y1-int(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
                cv2.putText(frame,"Potholes "+f"{n_detections}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_4)
                playsound(r'C:/Users/salla/OneDrive/Desktop/major data/zip/center.mp3')
        
    return frame



# calling main function to run the program

main(video_path=r"C:/Users/salla/OneDrive/Desktop/major data/zip/1.mp4", video_out="demo_result1.mp4") # Activate this fucntion when you pass any custom video
# main(webcam=True, video_out="demo_result.mp4") # Activate this fucntion when you need to capture output from webcam
# main(image_path="train_images/1.jpg") # Activate this fucntion when you need to capture output from image

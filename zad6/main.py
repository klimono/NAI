"""
Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

The program is designed to detect moving faces and shot them (by drawing a target on them.)
"""
import cv2 as cv
import numpy as np
import mediapipe as mp
import math

""" 
Initialization of a face cascade classifier with pre-trained XML file.
"""
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

"""
Function that draws sight on moving target.
"""
def shot(current_pose):
    """
    Load target coordinates from pose landmark (nose).
    """
    X = current_pose.landmark[0].x
    Y = current_pose.landmark[0].y
    
    x = int(-1080*Y)-size
    y = int(-1920*X)-size
    
    """
    Draw sight on target.
    """
    roi = frame[-size-x:-x, -size-y:-y] 
    roi[np.where(mask)] = 0
    roi += sight 
    return 0

"""
Function that detects movement.
"""
def detect_movement(prev_pose, current_pose):
    threshold = 0.005
    #print(current_pose.landmark[0])
    distY = (prev_pose.landmark[0].y - current_pose.landmark[0].y) ** 2
    distX = (prev_pose.landmark[0].x - current_pose.landmark[0].x) ** 2
    dist = math.sqrt(distX + distY)
    
    """
    If movement detected, call shot function.
    """
    if abs(dist > threshold) :
        #print(dist)
        shot(current_pose)
        return(True)
        
"""
Define webcam operator, pose variable and drawing utility. 
"""
cap = cv.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
drawing = mp.solutions.drawing_utils

"""
Load sight image.
"""
sight = cv.imread('sight.png')
size = 100
sight = cv.resize(sight, (size, size))

prev_pose = None


""" 
Main loop
each iteration = one frame
"""
while True:
    """
    Initialize frame, mask and sight variables. 
    """
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_sight = cv.cvtColor(sight, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray_sight, 1, 255, cv.THRESH_BINARY) 
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    current_pose = pose.process(rgb)
    frame = cv.blur(frame, (5, 5))
    
    """
    Draw exactly pose marks.
    """
    #drawing.draw_landmarks(frame, current_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    """
    Detect face and assign to variable
    """
    faces = face_cascade.detectMultiScale(gray, 1.1, 11 )
    
    """
    Check detected face movement.
    """
    if prev_pose is not None:    
        """
        If object moves, draw red rectangle on it. If it is stationary, draw green rectangle on it.
        """
        if(detect_movement(prev_pose, current_pose.pose_landmarks)):
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        else:
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    
    """
    Remember previous pose location.
    """
    prev_pose = current_pose.pose_landmarks
    
    """
    Show current frame.
    """
    cv.imshow("stream", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

""" 
Clean variables and destroy windows.
"""
cap.release()
cv.destroyAllWindows()

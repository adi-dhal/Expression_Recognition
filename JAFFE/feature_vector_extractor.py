import os
import dlib
import cv2
from random import sample
import numpy as np
import math
from scipy.spatial import distance

path = "./dataset2"

classes = ["AN","SA","HA","NE"]

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

files = sample(os.listdir(path),len(os.listdir(path)))

def eye_open(x,y):
	V_1 = distance.euclidean([x[1],y[1]],[x[5],y[5]])
	V_2 = distance.euclidean([x[2],y[2]],[x[4],y[4]])
	H_1 = distance.euclidean([x[3],y[3]],[x[0],y[0]])
	return (V_1+V_2)/(2.0*H_1)
	
def mouth_open(x,y):
	V_1 = distance.euclidean([x[1],y[1]],[x[7],y[7]])
	V_2 = distance.euclidean([x[2],y[2]],[x[6],y[6]])
	V_3 = distance.euclidean([x[3],y[3]],[x[5],y[5]])
	H_1 = distance.euclidean([x[4],y[4]],[x[0],y[0]])	
	return (V_1+V_2+V_3)/(3.0*H_1)

with open("feature_vector.txt",'w') as fl:
	for cnt,img_path in enumerate(files):
		class_name = img_path.split(".")[1][0:2]
		print class_name
		if class_name not in classes:
			continue
		img = cv2.imread(path + "/" +img_path , 0)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		img = clahe.apply(img)
		faces = face_detector(img,1)
		if len(faces) == 0:
			continue
		landmarks = landmark_detector(img,faces[0])
		x_coord = [float(landmarks.part(i).x) for i in range(68)]
		y_coord = [float(landmarks.part(i).y) for i in range(68)]
		x_mean = np.mean(x_coord)
		y_mean = np.mean(y_coord)
		vector = []
		angle_vector = []
		extra = []
		print len(x_coord)
		for i in range(68):				
			vector.append(distance.euclidean([x_mean,y_mean],[x_coord[i],y_coord[i]]))
			vector.append((math.atan2((y_coord[i] - y_mean), (x_coord[i] - x_mean))*360)/(2*math.pi))
		extra.append(eye_open(x_coord[36:42],y_coord[36:42]))
		extra.append(eye_open(x_coord[42:48],y_coord[36:42]))
		extra.append(mouth_open(x_coord[60:68],y_coord[60:68]))
		out = ""
		for i in vector:
			out = out + str(i) + ","
		out = out + str(extra[0])+"," + str(extra[1]) + ","+str(extra[2]) +"_"
		lab = [0,0,0,0]#[angry,sad,happy,neutral]
		lab[classes.index(class_name)] = 1
		out = out + str(lab[0])+ ","+ str(lab[1]) + ","+str(lab[2])+"," + str(lab[3])+"\n"
		fl.write(out)


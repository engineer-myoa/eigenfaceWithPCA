import matplotlib.pyplot as plt

import imghdr
import numpy as np
import os
import random
import cv2
from pprint import pprint


# ===================================
# GLOBAL AREA
# ===================================
IS_DEBUG = False

PATH_DATA = "train_data"
PATH_OUTPUT_EIGENFACES = "output_eigenfaces\\"
OUTPUT_EXTENSION = ".jpg"

# LABEL_DATA의 key는 train_data 밑 directory에 해당, value는 인식에 사용할 label string
LABEL_DATA  = { "label0" : "Unknown",
                "label1" : "person1",
                "label2" : "person2",
                "label3" : "person3",}
                # "label4" : "Soon-dragon"}
                
# ===================================
# CLASS DEFINE AREA
# ===================================

class faceRecogUsingPCA():

    numTrainingFaces = 56  # (all -91)  used
    eigvalThreshold = 0.99  # 0.95

    IMG_HEIGHT = IMG_ROW = None
    IMG_WIDTH = IMG_COLUMN = None


    def __init__(self, w, h, _eigvalThreshold=None):
        self.IMG_HEIGHT = self.IMG_ROW = h
        self.IMG_WIDTH = self.IMG_COLUMN = w

        self.eigvalThreshold = _eigvalThreshold

        self.initTrainingImage(PATH_DATA)

        self.faceCascade = cv2.CascadeClassifier()
        self.faceCascade.load("haarcascade_xml/haarcascade_frontalface_alt.xml")


    def initTrainingImage(self, path):

        def enumerateImagePaths(path_data):
            filenames = []
            for root, _, files in os.walk(path_data):
                path = root.split('/')
                for f in files:
                    filename = os.path.join(root, f)
                    if imghdr.what(filename): # 모듈없이 직접 파싱해도 가능하다
                        filenames.append(filename)
            return filenames


        trains = []
        labels = []

        """
        filenames = enumerateImagePaths(path)
        trains_data = filenames
        for name in trains_data:

            # 이미지 로드 및 Nx1 벡터로 변형
            tmpMat = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
            tmpMat = cv2.resize(tmpMat, (self.IMG_HEIGHT, self.IMG_WIDTH))
            tmpMat = tmpMat.reshape(-1, 1)

            # 트레이닝 이미지로 스태이킹
            try:
                trains = np.hstack((trains, tmpMat))
            except:
                trains = tmpMat

            # 레이블 데이터 추가. 트레이닝 인덱스 변경시 얘도 같이 따라다녀야함
            label = name.split("\\")[1]
            labels.append(label)
            
        """

        # trains_data = random.sample(filenames, numTrainingFaces)
        # trains_data = filenames[:numTrainingFaces] # only for dev

        dump=os.listdir("train_data")
        master_path = "train_data\\"
        for label_name in dump:
            full_path = master_path + label_name + "\\"
            train_files = os.listdir(full_path)
            for train_name in train_files:
                labels.append( label_name)

                tmpMat = cv2.imread(full_path + train_name, cv2.IMREAD_GRAYSCALE)
                tmpMat = cv2.resize(tmpMat, (self.IMG_HEIGHT, self.IMG_WIDTH))
                tmpMat = tmpMat.reshape(-1, 1)

                # 트레이닝 이미지로 스태이킹
                try:
                    trains = np.hstack((trains, tmpMat))
                except:
                    trains = tmpMat



        self.trains = trains
        self.labels = labels

    def train(self, _eigvalThreshold = None):

        if _eigvalThreshold != None:
            self.eigvalThreshold = _eigvalThreshold

        self.trains_mean = np.mean(np.array(self.trains), axis=1).reshape((-1, 1))
        self.trains_subMean = self.trains - self.trains_mean
        self.labels = np.array(self.labels)

        self.A_new = self.trains_subMean
        self.C_new = np.dot(self.A_new.T, self.A_new)
        self.eigvals, self.eigvecs = np.linalg.eig(self.C_new)

        # 고유값 크기 순서대로 재정렬을 위함
        indices = self.eigvals.argsort()[::-1]  # 값의 (-1이므로 큰) 순서대로 정렬하되, 그 순서의 index값을 배열로 리턴해줌
        self.eigvals = self.eigvals[indices]
        self.eigvecs = self.eigvecs[:, indices]
        self.labels = self.labels[indices]

        # =============================================

        eigvalSum = 0.0
        self.idxThreshold = -1
        sumOfEigvals = sum(self.eigvals)

        for idx, eigval in enumerate(self.eigvals):
            eigvalSum += eigval
            if eigvalSum / sumOfEigvals >= self.eigvalThreshold:
                self.idxThreshold = idx
                break

        print("의미있는 eigvals는 idx: {0} 까지".format(self.idxThreshold))

        self.U_new = None  # U_new는 eigenfaces의 집합임
        for i in range(self.idxThreshold):
            # K dims * idxThreshold 사이즈의 새로운 U행렬을 만듬
            # 여기서는 (2304,50) * (50,1) -> (2304,1) 이걸 앞의 25개만큼 horizontal(x축) 방향으로 stack
            try:
                self.U_new = np.hstack((self.U_new, np.dot(self.A_new, self.eigvecs.T[i].reshape(-1, 1))))  # 같은 모양이지만 명시적으로 (n,)에서 (n,1)로 바꿔줘야 제대로 hstack(x축)이 된다.
            except:
                self.U_new = np.dot(self.A_new, self.eigvecs.T[i].reshape(-1, 1))

        self.trains_weight = []
        for image_subMean in self.trains_subMean.T:
            self.weights = []

            for i in range(self.idxThreshold):
                wj = np.dot(self.U_new.T[i].reshape(1, -1), image_subMean.reshape(-1, 1))
                # uj = U_new[i]
                self.weights.append(wj.flatten()[0])
            faceReconstruction = np.zeros((self.IMG_WIDTH * self.IMG_HEIGHT, 1))

            for idx, weight in enumerate(self.weights):
                faceReconstruction += weight * self.U_new.T[idx].reshape(-1, 1)

            self.weights = np.array(self.weights)
            self.trains_weight.append(self.weights)

    def predict(self, _tmpMat):
        tmpMat = cv2.cvtColor(_tmpMat, cv2.COLOR_BGR2GRAY)
        tmpMat = cv2.resize(tmpMat, (self.IMG_HEIGHT, self.IMG_WIDTH))
        tmpMat = tmpMat.reshape(-1, 1)

        image_subMean = tmpMat - self.trains_mean
        weights = []

        for i in range(self.idxThreshold):
            wj = np.dot(self.U_new.T[i].reshape(1, -1), image_subMean.reshape(-1, 1))
            # uj = U_new[i]
            weights.append(wj.flatten()[0])

        weights = np.array(weights)

        minEuclideanDist = 10 ** 100
        minIdx = -1
        for idx in range(len(self.trains_weight)):
            tmpEuclideanDist = np.sum(np.power(self.trains_weight[idx] - weights, 2))
            if (minEuclideanDist > tmpEuclideanDist):
                minIdx = idx
                minEuclideanDist = tmpEuclideanDist
        whoami_label = self.labels[minIdx]
        whoami = LABEL_DATA.get(whoami_label)
        print(str(minIdx) + " / " + whoami + " / " + str(minEuclideanDist ))
        return whoami

    def detectFace(self, frame):
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vcRects = self.faceCascade.detectMultiScale(frame_bw, 1.05, 4)
        if (len(vcRects) == 0):
            return None, []

        roi = []
        for i in vcRects:
            x, y, w, h = i
            roi.append(frame[y:y + h, x:x + w])

        return roi, vcRects


# ===================================
# OPERATE AREA
# ===================================

if __name__ == "__main__":

    recog = faceRecogUsingPCA(48, 48, 0.99)
    recog.train()

    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        print("failed")
        exit(-1)
    # print(vc.get(3), vc.get(4))

    frame = None
    while(True):
        ret, frame = vc.read()


        rois, rects = recog.detectFace(frame)
        if len(rects) > 0:
            for idx, rect in enumerate(rects):

            #predict(roi)
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                name = recog.predict(rois[idx])
                # print(name, rect)
                cv2.putText(frame, name, (x+5,y+10), 2, 0.4, (0,255,0), 1)
        cv2.imshow("scene", frame)
        if cv2.waitKey(15) == 27: break


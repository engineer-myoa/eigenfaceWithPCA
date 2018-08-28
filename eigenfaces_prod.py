# coding=utf-8
import matplotlib.pyplot as plt

import argparse
import imghdr
import numpy as np
import os
import random
import cv2
from pprint import pprint


IS_DEBUG = False
IMG_HEIGHT = IMG_ROW    = 48
IMG_WIDTH  = IMG_COLUMN = 48

PATH_DATA = "train_data"
PATH_OUTPUT_EIGENFACES = "output_eigenfaces\\"
OUTPUT_EXTENSION = ".jpg"

LABEL_DATA  = { "label0" : "Unknown",
                "label1" : "Jung-ki",
                "label2" : "Kang",
                "label3" : "Woo-jin",
                "label4" : "Soon-dragon"}

numTrainingFaces = 36 # (all -91)  used
variance         = 0.99

if variance > 1.0:
    variance = 1.0
elif variance < 0.0:
    variance = 0.0

def enumerateImagePaths(root):
    filenames = []
    for root, _, files in os.walk(PATH_DATA):
        path = root.split('/')
        for f in files:
            filename = os.path.join(root, f)
            if imghdr.what(filename):
                filenames.append(filename)
    return filenames





# =============================================
# =============================================
# ↓ ↓ ↓    Random choice & load Image   ↓ ↓ ↓
# =============================================
# =============================================

filenames          = enumerateImagePaths(PATH_DATA)
print(filenames)
print(len(filenames))
trains_data = random.sample(filenames, numTrainingFaces)
# trains_data = filenames[:numTrainingFaces] # only for dev

trains = []
trains_new = []
labels_new = []
for name in trains_data:

    # 이미지 로드 및 Nx1 벡터로 변형
    tmpMat = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    tmpMat = cv2.resize(tmpMat, (IMG_HEIGHT, IMG_WIDTH ))
    trains.append( tmpMat )
    tmpMat = tmpMat.reshape(-1,1)

    # 트레이닝 이미지로 스태이킹
    try:
        trains_new = np.hstack( (trains_new, tmpMat) )
    except:
        trains_new = tmpMat

    # 레이블 데이터 추가. 트레이닝 인덱스 변경시 얘도 같이 따라다녀야함
    label = name.split("\\")[1]
    labels_new.append(label)






# =============================================
# =============================================
# ↓ ↓ ↓ Calculate & subtract average face ↓ ↓ ↓
# =============================================
# =============================================

trains_mean_new = np.mean(np.array(trains_new), axis=1).reshape((-1,1))
trains_subMean_new = trains_new - trains_mean_new
labels_new = np.array(labels_new)








# =============================================
# =============================================
#   ↓ ↓ ↓   Calculate eigen & sorting  ↓ ↓ ↓
# =============================================
# =============================================

A_new = trains_subMean_new
C_new = np.dot(A_new.T, A_new)
eigvals_new, eigvecs_new = np.linalg.eig(C_new)

# 고유값 크기 순서대로 재정렬을 위함
indices = eigvals_new.argsort()[::-1] # 값의 (-1이므로 큰) 순서대로 정렬하되, 그 순서의 index값을 배열로 리턴해줌
eigvals_new = eigvals_new[indices]
eigvecs_new = eigvecs_new[: ,indices]
labels_new = labels_new[indices]






# =============================================
# =============================================
#   ↓ ↓ ↓   Extract meaning eigenvals  ↓ ↓ ↓
# =============================================
# =============================================

eigvalThreshold = 0.99 # 0.95
eigvalSum = 0.0
idxThreshold = -1
sumOfEigvals = sum(eigvals_new)

for idx, eigval in enumerate(eigvals_new):
    eigvalSum += eigval
    if eigvalSum / sumOfEigvals >= eigvalThreshold:
        idxThreshold = idx
        break


print("의미있는 eigvals는 idx: {0} 까지".format(idxThreshold))







# =============================================
# =============================================
#     ↓ ↓ ↓    Calculate eigenface    ↓ ↓ ↓
# =============================================
# =============================================

U_new = None # U_new는 eigenfaces의 집합임
for i in range(idxThreshold):
    # K dims * idxThreshold 사이즈의 새로운 U행렬을 만듬
    # 여기서는 (2304,50) * (50,1) -> (2304,1) 이걸 앞의 25개만큼 horizontal(x축) 방향으로 stack
    try:
        U_new = np.hstack( (U_new , np.dot(A_new, eigvecs_new.T[i].reshape(-1,1))) ) # 같은 모양이지만 명시적으로 (n,)에서 (n,1)로 바꿔줘야 제대로 hstack(x축)이 된다.
    except:
        U_new = np.dot(A_new, eigvecs_new.T[i].reshape(-1,1))


# =============================================
# =============================================
#   ↓ Extract eigenfaces for visualization  ↓
#   IF DO NOT WANT RUN THIS STEP, ASSIGN "IS_DEBUG = False"
# =============================================
# =============================================
IS_DEBUG = True
if IS_DEBUG or True:
    idx = 0
    for idx, eigface in enumerate(U_new.T):
        dest = eigface.reshape((IMG_HEIGHT, IMG_WIDTH))
        dest = cv2.normalize(dest, None, 0, 255, cv2.NORM_MINMAX)
        dest = dest.astype(np.uint8)

        whoami = LABEL_DATA.get(labels_new[idx])
        # cv2.imshow("test", dest)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(PATH_OUTPUT_EIGENFACES + str(idx) + "_" + whoami + OUTPUT_EXTENSION, dest)
        cv2.imwrite(PATH_OUTPUT_EIGENFACES + str(idx) + "_" + OUTPUT_EXTENSION, dest)
        idx +=1
    del(idx)




# =============================================
# =============================================
#   ↓  prepare predict using test image set  ↓
# =============================================
# =============================================

# 남은 사진중 기존 trains_data에 들어있지 않은 것아닌 것
tests_data = np.setdiff1d( filenames, trains_data )
tests_new = []
for name in tests_data:
    tmpMat = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    tmpMat = cv2.resize(tmpMat, (IMG_HEIGHT, IMG_WIDTH))
    tmpMat = tmpMat.reshape(-1,1)
    # 트레이닝 이미지로 스태이킹
    try:
        tests_new = np.hstack((tests_new, tmpMat))
    except:
        tests_new = tmpMat

# tests_mean = np.mean(np.array(tests_new), axis=1).reshape((-1,1))
tests_subMean_new = tests_new - trains_mean_new # test의 mean이 아닌 trains의 mean을 빼야함





# =============================================
# =============================================
#   ↓ ↓ ↓  predict with reconstructing   ↓ ↓ ↓
# =============================================
# =============================================


# transpose 하는 이유는 내적의 우측피연산자이므로 2304,1 형태로 나와야함
# image_subMean.reshape(-1, 1)를 하는 이유는 필요한 크기가 (2304,1)이지만 (2304,)로 나오기때문
for image_subMean in tests_subMean_new.T:
    weights = []

    for i in range(idxThreshold):

        wj = np.dot(U_new.T[i].reshape(1, -1), image_subMean.reshape(-1, 1))
        # uj = U_new[i]
        weights.append( wj.flatten()[0] )
    faceReconstruction = np.zeros( (IMG_WIDTH*IMG_HEIGHT, 1) )

    for idx, weight in enumerate(weights):
        faceReconstruction += weight * U_new.T[idx].reshape(-1,1)



    # f = plt.figure()
    # plt.suptitle("enhanced solution")
    #
    # f.add_subplot(1, 2, 1)
    # plt.imshow(faceReconstruction.reshape((IMG_HEIGHT, IMG_WIDTH)),cmap=plt.cm.Greys_r)
    # plt.title("reconstruction")
    #
    #
    # f.add_subplot(1, 2, 2)
    # plt.imshow(image_subMean.reshape((IMG_HEIGHT, IMG_WIDTH)),cmap=plt.cm.Greys_r)
    # plt.title("image")
    #
    # plt.show()





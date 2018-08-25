## Eigenface Recognizer with PCA in video

# Introduction
사물 및 얼굴 인식 기술은 개인을 인증하는 수단부터 디지털 포렌식까지 다양한 분야에서 활용될 수 있는 영상처리에서 중요하게 여기어지는 분야중 하나이다. 그 중 얼굴인식은 사람의 얼굴특성을 활용하여 얼굴을 감지한 뒤, 훈련된 이미지 정보들과 비교해 어느정도 일치하는지에 따라 얼굴을 인식한다. 얼굴 인식기는 그 방법에 따라 크게 eigen face, fisher face, lbph face로  나눌 수 있으며 본 프로젝트에서는 가장 기초가 되는 eigen face를 활용한 얼굴 인식기를 설계 및 구현한다.
Eigen face를 활용해 얼굴인식을 하기 위해서는 성능적 측면을 위해 고려되는 몇 가지 사항이 있다. 먼저 Eigen face 생성을 위해 공분산행렬, 고유값 계산이 필요하다. 이 과정에서 이미지 크기에 의존되는 차원의 연산이 필요한데, 공분산 행렬 계산 순서를 뒤집음으로 이미지 크기 대신 이미지 수에 의존되는 차원으로 사상하도록 바꾸어 연산 속도를 월등히 빠르게 높일 수 있다. 또한 고유벡터는 데이터를 특정 축에 사상하였을 때 분산 정도에 따라 해당 축의 중요성을 파악할 수 있는 특성을 가지는데 PCA(Principle Component Analyser)기법을 이용하여 에러를 최소화함과 동시에 필요없는 데이터를 절삭하도록 한다. 따라서 성능개선이 가능한 알고리즘들을 이용해 얼굴인식기를 구현한다.
구현 결과물을 통해 Eigen face를 통한 인식기 자체가 고성능의 인식기는 아니지만 PCA 기법을 이해하기 쉽고 유용한 예제로서 쓰일 수 있음을 알 수 있다. 따라서 본 프로젝트에서는 OpenCV Contribute 라이브러리로 제공되는 Face 모듈을 사용하지 않고 Numpy를 주로 이용해 PCA기법을 적용한 Eigen face 얼굴 인식기 구현을 목표로한다.

# How To Use It
Dependency Python Reqirements
```
pip install opencv-python

pip install numpy

pip install imghdr (Optional)

pip install matplotlib
```

train_data
```
$ mkdir ./train_data/some_label ...

...
...

$ ls
some_label1 some_label2 some_label3 ...

# and edit "LABEL_DATA" in facerecog_video.py
# folder names are dictionary's key, real recognization labels are value
```

Input Video Source
```
# just set up your webcam source like VideoCapture(0)
```


# References

https://github.com/Submanifold/Eigenfaces/blob/master/eigenfaces.py#L3

https://github.com/zwChan/Face-recognition-using-eigenfaces/blob/master/eigenFace.py

https://github.com/kevinhughes27/pyIPCA/blob/master/examples/eigenface.py

https://github.com/agyorev/Eigenfaces/blob/master/eigenfaces.py
# raspberry-mnist
라즈베리파이를 이용하여 손으로 쓴 숫자를 인식해보자

- ## Model Layer
    ![image](https://github.com/vkdnjznd/raspberry-mnist/blob/master/doc/model.jpg)
- ## Training Data

<center> Mnist Dataset </center>

![image](https://github.com/vkdnjznd/raspberry-mnist/blob/master/doc/MnistExamples.png)

시스템 환경은 다음과 같다 
- Model = Raspberry Pi 4 4GB
- OS = Raspbian
- Language = Python 3.7.3
#
## Setup
#### Step 1 라즈비안 업데이트
    sudo apt-get update
    sudo apt-get dist-upgrade
***
#### Step 2 저장소 클론
    mkdir raspberry_mnist
    pyvenv raspberry_mnist
    cd raspberry_mnist

    git clone https://github.com/vkdnjznd/raspberry-mnist.git
---
#### Step 3 라즈비안에 패키지 설치
## <center>카메라 설정이 Enable 인지 확인</center>
![image](https://github.com/vkdnjznd/raspberry-mnist/blob/master/doc/camera.jpg)

가상환경(virtualenv)에 설치하는것을 권장
    
    source bin/activate
    bash get_pi_requirements.sh

약 400MB 정도의 필요 라이브러리를 자동으로 설치함.

출처 : https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/get_pi_requirements.sh

***
#### Step 4 실행
    python3 TFLite_mnist_webcam.py --modeldir=mnist_model
![image](https://github.com/vkdnjznd/raspberry-mnist/blob/master/doc/test.jpg)




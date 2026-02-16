import cv2
import os
import urllib.request


def download_file(url, filename):
    """파일 다운로드"""
    if os.path.exists(filename):
        print(f"{filename} 이미 존재합니다.")
        return
    
    print(f"{filename} 다운로드 중...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} 다운로드 완료!")
    except Exception as e:
        print(f"다운로드 실패: {e}")


def setup_opencv_models():
    """OpenCV 객체 탐지 모델 다운로드"""
    print("OpenCV 객체 탐지 모델 설정 중...")
    
    # MobileNet SSD 모델
    model_url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
    config_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    labels_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/classification_classes_ILSVRC2012.txt"
    
    # COCO 레이블
    coco_labels = """person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush"""
    
    # 레이블 파일 저장
    if not os.path.exists('Labels.txt'):
        with open('Labels.txt', 'w') as f:
            f.write(coco_labels)
        print("Labels.txt 생성 완료")
    
    print("\n설정 완료!")
    print("참고: YOLO 모델을 사용하려면 'pip install ultralytics' 후")
    print("첫 실행 시 자동으로 yolov8n.pt가 다운로드됩니다.")


def setup_directories():
    """필요한 디렉토리 생성"""
    dirs = ['known_faces', 'output', 'test_images']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"디렉토리 생성: {d}")


if __name__ == "__main__":
    print("=" * 50)
    print("🔧 환경 설정")
    print("=" * 50)
    
    setup_directories()
    setup_opencv_models()
    
    print("\n설정이 완료되었습니다!")
    print("이제 'python main.py'를 실행하세요.")
# Vision Program

Python + OpenCV를 사용한 얼굴 인식 및 객체 탐지 애플리케이션

## 기능

- **얼굴 인식**: 웹캠 또는 이미지에서 얼굴을 인식하고 등록된 사람 식별
- **객체 탐지**: YOLO 또는 OpenCV DNN을 사용한 실시간 객체 탐지
- **이미지 처리**: 이미지 파일에 대한 얼굴 인식 및 객체 탐지

## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

**참고**: Windows에서 dlib 설치가 어려울 경우:
```bash
pip install cmake
pip install dlib
```

### 2. 환경 설정

```bash
python setup.py
```

## 사용 방법

### 메인 프로그램 실행

```bash
python main.py
```

메뉴:
1. **웹캠 - 얼굴 인식**: 실시간 얼굴 인식 (등록된 얼굴 필요)
2. **웹캠 - 객체 탐지**: 실시간 객체 탐지
3. **이미지 파일 처리**: 이미지에서 얼굴/객체 탐지

### 얼굴 등록

```bash
python register_face.py
```

1. "새 얼굴 등록" 선택
2. 웹캠 앞에서 얼굴을 보여주고 's' 키로 촬영
3. 이름 입력
4. "등록된 얼굴 테스트"로 확인

## 파일 구조

```
visionProgram/
├── main.py              # 메인 애플리케이션
├── register_face.py     # 얼굴 등록 스크립트
├── setup.py            # 환경 설정
├── requirements.txt    # 패키지 목록
├── known_faces/        # 등록된 얼굴 이미지
├── output/             # 결과 저장
└── test_images/        # 테스트 이미지
```

## 단축키

- **q**: 종료
- **s**: 얼굴 등록 시 촬영

## 문제 해결

### 웹캠이 안 켜질 때
- 다른 프로그램에서 웹캠을 사용 중인지 확인
- 카메라 권한 설정 확인

### 얼굴 인식이 안 될 때
- 밝은 환경에서 시도
- 얼굴이 카메라에 정면으로 보이게 하기
- known_faces 폴더에 이미지가 있는지 확인

### YOLO 모델 다운로드
첫 실행 시 자동으로 다운로드됩니다. 수동 다운로드:
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 라이선스

MIT License
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime


class VisionApp:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.camera = None
        
    def add_known_face(self, image_path, name):
        """등록된 얼굴 추가"""
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        print(f"등록 완료: {name}")
        
    def recognize_faces(self, frame):
        """얼굴 인식 수행"""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            face_names.append(name)
            
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
        return frame
    
    def detect_objects_yolo(self, frame):
        """YOLO로 객체 탐지"""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            results = model(frame)
            annotated_frame = results[0].plot()
            return annotated_frame
        except ImportError:
            return self.detect_objects_opencv(frame)
    
    def detect_objects_opencv(self, frame):
        """OpenCV DNN으로 객체 탐지 (YOLO 없을 때)"""
        try:
            config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
            frozen_model = 'frozen_inference_graph.pb'
            
            if not os.path.exists(config_file) or not os.path.exists(frozen_model):
                cv2.putText(frame, "Model files not found. Run setup_models.py", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame
            
            model = cv2.dnn_DetectionModel(frozen_model, config_file)
            model.setInputSize(320, 320)
            model.setInputScale(1.0/127.5)
            model.setInputMean((127.5, 127.5, 127.5))
            model.setInputSwapRB(True)
            
            classLabels = []
            file_name = 'Labels.txt'
            with open(file_name, 'rt') as fpt:
                classLabels = fpt.read().rstrip('\n').split('\n')
            
            ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
            
            if len(ClassIndex) != 0:
                for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                    if ClassInd <= 80:
                        cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                        cv2.putText(frame, classLabels[ClassInd-1], 
                                   (boxes[0]+10, boxes[1]+40), 
                                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, f"Object detection error: {str(e)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def start_webcam(self, mode='face'):
        """웹캠 시작
        mode: 'face' = 얼굴 인식, 'object' = 객체 탐지
        """
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("웹캠을 찾을 수 없습니다!")
            return
        
        print(f"웹캠 시작 - 모드: {mode}")
        print("종료하려면 'q'를 누르세요")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            if mode == 'face':
                frame = self.recognize_faces(frame)
            elif mode == 'object':
                frame = self.detect_objects_yolo(frame)
            
            cv2.imshow('Vision App', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path, mode='face', output_path=None):
        """이미지 파일 처리"""
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return
        
        if mode == 'face':
            frame = self.recognize_faces(frame)
        elif mode == 'object':
            frame = self.detect_objects_yolo(frame)
        
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"결과 저장: {output_path}")
        
        cv2.imshow('Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VisionApp()
    
    print("=" * 50)
    print("🎥 비전 프로그램")
    print("=" * 50)
    print("1. 웹캠 - 얼굴 인식")
    print("2. 웹캠 - 객체 탐지")
    print("3. 이미지 파일 처리")
    print("4. 종료")
    print("=" * 50)
    
    choice = input("선택: ")
    
    if choice == '1':
        app.start_webcam(mode='face')
    elif choice == '2':
        app.start_webcam(mode='object')
    elif choice == '3':
        image_path = input("이미지 경로: ")
        mode = input("모드 (face/object): ")
        app.process_image(image_path, mode=mode)
    else:
        print("종료합니다.")
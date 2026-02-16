import cv2
import os
from main import VisionApp


def capture_face_for_registration():
    """웹캠으로 얼굴 캡처 및 등록"""
    print("얼굴 등록을 시작합니다...")
    print("웹캠 앞에서 얼굴을 정면으로 보여주세요.")
    print("촬영하려면 's'를 누르세요.")
    print("취소하려면 'q'를 누르세요.")
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("웹캠을 찾을 수 없습니다!")
        return
    
    # known_faces 폴터 생성
    os.makedirs('known_faces', exist_ok=True)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # 얼굴 영역 표시 (Dlib이 필요하므로 간단히 사각형만)
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        box_size = 300
        
        top_left = (center_x - box_size//2, center_y - box_size//2)
        bottom_right = (center_x + box_size//2, center_y + box_size//2)
        
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, "Face Registration Area", (top_left[0], top_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 얼굴 영역만 저장
            face_area = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            name = input("이름을 입력하세요: ")
            filename = f"known_faces/{name}.jpg"
            cv2.imwrite(filename, face_area)
            print(f"저장 완료: {filename}")
            
            add_more = input("더 등록하시겠습니까? (y/n): ")
            if add_more.lower() != 'y':
                break
                
        elif key == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()
    print("등록이 완료되었습니다!")


def test_face_recognition():
    """등록된 얼굴로 테스트"""
    app = VisionApp()
    
    # known_faces 폴터에서 모든 이미지 로드
    known_faces_dir = 'known_faces'
    
    if not os.path.exists(known_faces_dir):
        print("등록된 얼굴이 없습니다. 먼저 register_face.py를 실행하세요.")
        return
    
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, filename)
            name = os.path.splitext(filename)[0]
            try:
                app.add_known_face(image_path, name)
            except Exception as e:
                print(f"{name} 로드 실패: {e}")
    
    if not app.known_face_names:
        print("등록된 얼굴이 없습니다!")
        return
    
    print(f"\n등록된 인물: {', '.join(app.known_face_names)}")
    print("\n웹캠 테스트를 시작합니다...")
    app.start_webcam(mode='face')


if __name__ == "__main__":
    print("=" * 50)
    print("👤 얼굴 등록 및 테스트")
    print("=" * 50)
    print("1. 새 얼굴 등록")
    print("2. 등록된 얼굴 테스트")
    print("3. 종료")
    print("=" * 50)
    
    choice = input("선택: ")
    
    if choice == '1':
        capture_face_for_registration()
    elif choice == '2':
        test_face_recognition()
    else:
        print("종료합니다.")
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from datetime import datetime

st.set_page_config(
    page_title="Vision Web App",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Vision Web App")
st.markdown("### 얼굴 인식 & 객체 탐지 웹앱")

# 세션 상태 초기화
if 'known_faces' not in st.session_state:
    st.session_state.known_faces = {}

# 사이드바 메뉴
menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["🏠 홈", "📹 실시간 스트리밍(webrtc)", "📸 웹캠(스냅샷)", "🖼️ 이미지 분석", "👤 얼굴 등록", "📚 사용법"]
)

def detect_faces_opencv(image):
    """OpenCV로 얼굴 탐지"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, len(faces)

def detect_objects_opencv(image):
    """OpenCV DNN으로 객체 탐지"""
    try:
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'
        
        if os.path.exists(config_file) and os.path.exists(frozen_model):
            model = cv2.dnn_DetectionModel(frozen_model, config_file)
            model.setInputSize(320, 320)
            model.setInputScale(1.0/127.5)
            model.setInputMean((127.5, 127.5, 127.5))
            model.setInputSwapRB(True)
            
            class_labels = []
            if os.path.exists('Labels.txt'):
                with open('Labels.txt', 'rt') as f:
                    class_labels = f.read().rstrip('\n').split('\n')
            
            ClassIndex, confidence, bbox = model.detect(image, confThreshold=0.5)
            
            if len(ClassIndex) != 0:
                for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                    if ClassInd <= 80 and len(class_labels) >= ClassInd:
                        cv2.rectangle(image, boxes, (255, 0, 0), 2)
                        label = class_labels[ClassInd-1] if len(class_labels) >= ClassInd else f"Class {ClassInd}"
                        cv2.putText(image, label, 
                                   (boxes[0]+10, boxes[1]+40), 
                                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            st.warning("객체 탐지 모델 파일이 없습니다.")
    except Exception as e:
        st.error(f"객체 탐지 오류: {e}")
    
    return image

if menu == "🏠 홈":
    st.markdown("""
    ## 환영합니다! 👋
    
    이 웹앱은 **OpenCV**를 사용하여 다음 기능을 제공합니다:
    
    ### ✨ 주요 기능
    - 📸 **실시간 웹캠**: 친구 얼굴 탐지 및 객체 인식
    - 🖼️ **이미지 분석**: 업로드한 사진 분석
    - 👤 **얼굴 등록**: 새로운 얼굴 등록 및 관리
    
    ### 📱 어디서든 사용 가능
    - PC, 태블릿, 스마트폰 모두 지원
    - 웹 브라우저만 있으면 OK!
    
    ### 🚀 시작하기
    왼쪽 사이드바에서 원하는 기능을 선택하세요!
    """)

elif menu == "📹 실시간 스트리밍(webrtc)":
    st.header("📹 실시간 스트리밍(webrtc)")
    st.caption("브라우저 카메라 스트림을 받아서 실시간으로 처리해. (진짜 실시간)")

    # streamlit-webrtc는 import 비용이 좀 있어서 여기서 import
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av

    mode = st.radio("모드 선택", ["얼굴 탐지", "객체 탐지"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        conf = st.slider("객체 탐지 confThreshold", 0.1, 0.9, 0.5, 0.05)
    with col2:
        face_scale = st.slider("얼굴 탐지 scaleFactor", 1.05, 1.5, 1.10, 0.01)

    class Processor(VideoProcessorBase):
        def __init__(self):
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            if mode == "얼굴 탐지":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, face_scale, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, f"faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                # 기존 함수는 파일 존재여부 등 Streamlit UI에 경고를 띄우기 때문에
                # 실시간에서는 아주 단순히 '모델파일 있으면'만 처리
                try:
                    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
                    frozen_model = 'frozen_inference_graph.pb'
                    if os.path.exists(config_file) and os.path.exists(frozen_model):
                        model = cv2.dnn_DetectionModel(frozen_model, config_file)
                        model.setInputSize(320, 320)
                        model.setInputScale(1.0/127.5)
                        model.setInputMean((127.5, 127.5, 127.5))
                        model.setInputSwapRB(True)
                        ClassIndex, confidence, bbox = model.detect(img, confThreshold=float(conf))
                        if len(ClassIndex) != 0:
                            for ClassInd, c, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                                cv2.rectangle(img, boxes, (0, 255, 0), 2)
                                cv2.putText(img, f"{int(ClassInd)} {c:.2f}", (boxes[0], max(0, boxes[1]-10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, "(object model missing)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                except Exception:
                    cv2.putText(img, "object detect error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="realtime",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=Processor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
            ]
        },
        video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True},
        desired_playing_state=True,
        async_processing=True,
    )

    st.info("카메라 권한 허용하면 바로 실시간으로 따라가. 끊기면 새로고침(F5)하면 돼.")

elif menu == "📸 웹캠(스냅샷)":
    st.header("📸 웹캠(스냅샷)")
    
    mode = st.radio("모드 선택", ["얼굴 탐지", "객체 탐지"])
    
    # 웹캠 입력
    camera_image = st.camera_input("웹캠으로 사진 찍기")
    
    if camera_image is not None:
        # 이미지 처리
        bytes_data = camera_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        if mode == "얼굴 탐지":
            result_img, face_count = detect_faces_opencv(cv2_img)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                    caption=f"탐지된 얼굴: {face_count}개")
        else:
            result_img = detect_objects_opencv(cv2_img)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                    caption="객체 탐지 결과")
        
        # 결과 저장
        if st.button("💾 결과 저장"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/result_{timestamp}.jpg"
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(filename, result_img)
            st.success(f"저장 완료: {filename}")

elif menu == "🖼️ 이미지 분석":
    st.header("🖼️ 이미지 분석")
    
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="원본 이미지", use_column_width=True)
        
        # 분석 모드 선택
        analysis_mode = st.radio("분석 모드", ["얼굴 탐지", "객체 탐지"])
        
        if st.button("🔍 분석 시작"):
            with st.spinner("분석 중..."):
                # OpenCV용으로 변환
                img_array = np.array(image)
                cv2_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                if analysis_mode == "얼굴 탐지":
                    result_img, face_count = detect_faces_opencv(cv2_img)
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                            caption=f"탐지된 얼굴: {face_count}개")
                    
                    if face_count > 0:
                        st.success(f"✅ {face_count}개의 얼굴을 찾았습니다!")
                    else:
                        st.warning("얼굴을 찾을 수 없습니다.")
                else:
                    result_img = detect_objects_opencv(cv2_img)
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                            caption="객체 탐지 결과")

elif menu == "👤 얼굴 등록":
    st.header("👤 얼굴 등록")
    
    st.markdown("""
    새로운 얼굴을 등록합니다. 웹캠으로 촬영하거나 이미지를 업로드하세요.
    """)
    
    name = st.text_input("이름 입력")
    
    register_method = st.radio("등록 방법", ["웹캠으로 촬영", "이미지 업로드"])
    
    if register_method == "웹캠으로 촬영":
        camera_image = st.camera_input("얼굴을 보여주고 촬영하세요")
        if camera_image is not None and name:
            if st.button("등록하기"):
                os.makedirs("known_faces", exist_ok=True)
                with open(f"known_faces/{name}.jpg", "wb") as f:
                    f.write(camera_image.getvalue())
                st.success(f"✅ {name}님이 등록되었습니다!")
    else:
        uploaded_file = st.file_uploader("얼굴 사진 업로드", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None and name:
            if st.button("등록하기"):
                os.makedirs("known_faces", exist_ok=True)
                with open(f"known_faces/{name}.jpg", "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.success(f"✅ {name}님이 등록되었습니다!")
    
    # 등록된 얼굴 목록
    st.subheader("등록된 얼굴 목록")
    if os.path.exists("known_faces"):
        faces = os.listdir("known_faces")
        if faces:
            for face in faces:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"👤 {face.replace('.jpg', '').replace('.png', '')}")
                with col2:
                    if st.button("삭제", key=face):
                        os.remove(f"known_faces/{face}")
                        st.experimental_rerun()
        else:
            st.info("등록된 얼굴이 없습니다.")

elif menu == "📚 사용법":
    st.header("📚 사용법")
    
    st.markdown("""
    ### 🎯 빠른 시작
    
    1. **📸 실시간 웹캠**
       - '웹캠으로 사진 찍기' 버튼 클릭
       - 얼굴이나 객체를 카메라에 보여주세요
       - 자동으로 탐지됩니다!
    
    2. **🖼️ 이미지 분석**
       - 분석할 사진을 업로드
       - '분석 시작' 버튼 클릭
       - 결과 확인!
    
    3. **👤 얼굴 등록**
       - 이름 입력
       - 웹캠 또는 이미지로 등록
       - 등록된 얼굴은 목록에서 관리
    
    ### 💡 팁
    - 밝은 곳에서 사용하면 더 잘 인식됩니다
    - 얼굴은 정면으로 보여주세요
    - 여러 사람이 있어도 모두 탐지됩니다
    
    ### 🔒 개인정보 보호
    - 모든 데이터는 로컬에 저장됩니다
    - 외부 서버로 전송되지 않습니다
    """)

# 푸터
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit & OpenCV")

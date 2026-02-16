# Vision Web App

Streamlit으로 만든 얼굴 인식 & 객체 탐지 웹앱

## 빠른 시작

### 1. 패키지 설치
```bash
pip install -r requirements_web.txt
```

### 2. 웹앱 실행
```bash
streamlit run app.py
```

### 3. 접속
- 자동으로 브라우저가 열립니다
- 또는 http://localhost:8501 접속

## 배포 방법

### 옵션 1: Streamlit Cloud (묣료) - 추천

1. GitHub에 코드 업로드
2. https://streamlit.io/cloud 접속
3. GitHub 계정 연동
4. "New app" → 저장소 선택
5. Deploy!

**장점:**
- 완전 묣료
- 자동 HTTPS
- 핸드폰/태블릿/PC 어디서든 접속
- 업데이트하면 자동 배포

### 옵션 2: 로컬 네트워크 (테스트용)

같은 와이파이 내에서 핸드폰으로 접속:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

그런 다음 핸드폰 브라우저에서:
```
http://[컴퓨터IP]:8501
```

### 옵션 3: Heroku (묣료)

1. Heroku CLI 설치
2. Procfile 생성:
   ```
   web: streamlit run app.py --server.port $PORT
   ```
3. 배포:
   ```bash
   git push heroku main
   ```

## 사용 방법

1. **홈**: 앱 소개
2. **실시간 웹캠**: 친구 얼굴/객체 실시간 탐지
3. **이미지 분석**: 사진 업로드하여 분석
4. **얼굴 등록**: 새 얼굴 등록 및 관리

## 파일 구조
```
visionProgram/
├── app.py                 # Streamlit 웹앱
├── requirements_web.txt   # 웹앱 패키지
└── README_WEB.md         # 이 파일
```
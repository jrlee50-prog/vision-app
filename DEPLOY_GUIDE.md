# 🚀 배포 완전 가이드

## 단계 1: GitHub에 코드 업로드 (5분)

### 1.1 GitHub 계정 만들기
1. https://github.com 에 접속
2. "Sign up" 클릭
3. 이메일, 비밀번호, 사용자명 입력
4. 이메일 인증 완료

### 1.2 새 저장소(Repository) 만들기
1. GitHub 로그인 후 오른쪽 상단 + 버튼 클릭
2. "New repository" 선택
3. 저장소 이름 입력: `vision-web-app`
4. "Create repository" 클릭
5. 생성된 저장소 URL 복사 (예: `https://github.com/사용자명/vision-web-app`)

### 1.3 코드 업로드하는 2가지 방법

#### 방법 A: GitHub 웹사이트에서 직접 업로드 (쉬움)
1. 저장소 페이지에서 "Add file" → "Upload files" 클릭
2. "choose your files" 클릭
3. 다음 파일들 선택:
   - app.py
   - requirements_web.txt
   - Labels.txt (있는 경우)
4. "Commit changes" 클릭

#### 방법 B: Git 명령어 사용 (Git 설치 필요)
```bash
# Git 설치 확인
git --version

# 폴터 이동
cd C:\Users\Jerry\Desktop\visionProgram

# Git 초기화
git init

# 파일 추가
git add app.py requirements_web.txt Labels.txt

# 커밋
git commit -m "Initial commit"

# GitHub 저장소 연결 (방금 만든 저장소 URL 사용)
git remote add origin https://github.com/사용자명/vision-web-app.git

# 업로드
git push -u origin main
```

**참고**: Windows에서 Git이 없다면 https://git-scm.com/download/win 에서 설치

---

## 단계 2: Streamlit Cloud 배포 (3분)

### 2.1 Streamlit Cloud 가입
1. https://streamlit.io/cloud 접속
2. "Sign up with GitHub" 클릭
3. GitHub 계정으로 로그인
4. 권한 요청 → "Authorize streamlit" 클릭

### 2.2 앱 배포하기
1. Streamlit Cloud 대시보드에서 "New app" 클릭
2. "Deploy an app from a GitHub repository" 선택
3. 설정 입력:
   - **Repository**: 방금 만든 저장소 선택 (`사용자명/vision-web-app`)
   - **Branch**: `main` 또는 `master`
   - **Main file path**: `app.py`
   - **App URL**: 원하는 이름 입력 (예: `my-vision-app`)
4. "Advanced settings..." 클릭:
   - **Python version**: 3.9 선택
5. "Deploy!" 클릭

### 2.3 배포 완료!
- 앱이 빌드되는 동안 기다림 (2-3분)
- 성공하면 URL이 생성됨: `https://my-vision-app.streamlit.app`
- 이 URL을 핸드폰/태블릿/PC 브라우저에서 열 수 있음!

---

## 단계 3: 모바일에서 사용하기

### iPhone/iPad
1. Safari 또는 Chrome 열기
2. 배포된 URL 입력 (예: `https://my-vision-app.streamlit.app`)
3. "홈 화면에 추가" → 앱처럼 사용 가능!

### Android
1. Chrome 브라우저 열기
2. 배포된 URL 입력
3. 메뉴 → "홈 화면에 추가" → 바로가기 생성

---

## 문제 해결

### 배포 실패 시 확인사항
1. **requirements_web.txt** 파일이 저장소에 있는지 확인
2. **app.py** 파일명이 정확한지 확인
3. GitHub 저장소가 Public(공개)인지 확인

### "Module not found" 오류
requirements_web.txt에 필요한 패키지가 모두 있는지 확인:
```
streamlit==1.28.0
opencv-python==4.8.1.78
numpy==1.26.2
pillow==10.1.0
```

### 친구가 접속할 수 없어요
- URL을 정확히 공유했는지 확인
- 앱이 "Sleeping" 상태일 수 있음 (클릭하면 30초 후 실행됨)

---

## 업데이트 방법

코드를 수정한 후 다시 배포하려면:

### GitHub에서 직접 수정
1. 저장소 페이지에서 파일 클릭
2. 연필 아이콘(✏️) 클릭
3. 코드 수정
4. "Commit changes" 클릭
5. 자동으로 Streamlit Cloud가 업데이트됨!

### Git으로 업데이트
```bash
git add .
git commit -m "코드 업데이트"
git push
```

---

## 완료! 🎉

이제 당신만의 AI 비전 웹앱이 인터넷에 올라갔습니다!
- 핸드폰으로 접속해서 테스트필요
- 친구들에게 URL 공유 가능
- 24시간 실행됨 (묣료 플랜은 7일간 미사용 시 Sleep)

질문 있으시면 물어보세요!
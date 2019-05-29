# MakehardSub
영상 자체자막 추출 유틸리티

# 설치
## python 설치
  python이 설치되어 있지 않다면 아래 주소에서 python 3.7 또는 3.6을 다운로드 받아서 설치한다.
https://www.python.org/downloads/windows/

```
python --version 
```
위 명령어를 실행했을 때 python 버전이 출력되면 정상적으로 설치가 됨

## Tesseract 설치
https://github.com/UB-Mannheim/tesseract/wiki 에서 Tesseract 다운로드 후 설치

영어가 아닌 다른 언어는 학습 데이터를 다운로드 받아서 
tesseract가 설치된 디렉토리(예 C:\Program Files\Tesseract-OCR\tessdata)에 복사한다.  

https://github.com/tesseract-ocr/tessdata

Shell에서
```
where tesseract

```
명령어를 실행했을 때 tesseract 설치 경로가 나타나지 않는다면 재부팅해 본다.


## python 라이브러리 설치
pip install -r requirements.txt

# 사용법
## 자체 자막의 타임 코드 추출
```
python hardsub.py --video example.mp4
```
 
옵션  
--video: 영상 파일명  
--pos: 자막 영역의 좌표(left,top,width,height)  
--frame_window: 프레임 간격(ms 단위)  

실행이 끝나면 타임 코드 자막이 생성된다.(영상이 example.mp4이면 example.srt)

## 자체 자막의 자막 추출
```
python hardsub.py --video example.mp4 --ref example.srt --output example.smi --lang eng
```

타임코드 자막을 이용하여 해당 프레임에 있는 자막을 추출한다.
(Tesseract OCR를 이용해서 문자 인식)

옵션  
--video: 영상 파일명  
--pos: 자막 영역의 좌표(left,top,width,height)  
--ref: 타임코드를 가지고 있는 자막 파일명  
--lang: 자막의 언어(영어:eng, 한글:kor, 중국어:chi_sim 또는 chi_tra)  
--output: 출력 자막 파일명  
--ipm: OCR 인식을 위한 이미지 전처리 모드(기본값은 0: 전처리 안함)  


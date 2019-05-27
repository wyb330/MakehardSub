# MakehardSub
영상 자체자막 추출 유틸리티

# 설치
## Tesseract 설치
https://github.com/UB-Mannheim/tesseract/wiki 에서 Tesseract 다운로드 후 설치

영어가 아닌 다른 언어는 학습 데이터를 다운로드 받아서 
tesseract가 설치된 디렉토리(예 C:\Program Files\Tesseract-OCR\tessdata)에 복사한다.
https://github.com/tesseract-ocr/tessdata

## python 라이브러리 설치
pip install -r requirements.txt

# 사용법
## 자체 자막의 타임 코드 추출
python hardsub.py --video example.mp4
 
옵션
--video: 영상 파일명
--pos: 자막의 위치(left,top,wirdth,height)

실행이 끝나면 타임 코드 자막이 생성된다.(영상이 example.mp4이면 example.srt)

## 자체 자막의 자막 추출
python hardsub.py --video example.mp4 --ref example.srt --lang eng

옵션
--video: 영상 파일명
--pos: 자막의 위치(left,top,wirdth,height)
--ref: 타임코드를 가지고 있는 자막 파일명
--lang: 자막의 언어(영어:eng, 한글:kor, 중국어:chi_sim 또는 chi_tra)
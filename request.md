# 목적
- 이 코드는 효율적으로 VLM모델을 serving하기 위한 코드이다. 제공해야하는 서비스는 아래와 같다.
    - 요약문 생성
    - 행동인식
    - 물체탐지

## 서비스별 사용 모델
- 요약문 생성, 행동인식, 물체탐지 - Ovis2 8B

## 서비스별 구체내용
- 요약문 생성
    - 입력 : 연속된 이미지
    - 요약문을 생성하기에 앞서 입력이 들어온 연속된 이미지에 대해서 카테고리 분류를 진행한다. 
    - 카테고리 분류를 위한 프롬프트는 def category_prompt() 함수의 아웃풋으로 한다.
    - 카테고리 분류에서 나온 출력을 parsing하여 요약문 생성 프롬프트에 입력으로 들어간다
    - 요약문 생성 프롬프트는 def summarize_prompt()함수의 아웃풋으로 한다
    - 입력들어온 연속된 이미지와 summarize_prompt()의 출력을 Ovis2 8B 모델에 입력넣어서 생성된 출력을 response로 회신한다.
- 행동인식
    - 입력 : 연속된 이미지
    - 입력들어온 연속된 이미지와 def motion_recognition_prompt()함수의 아웃풋을 Ovis2 8B 모델에 입력으로 사용한다
    - 출력된 결과를 response로 회신한다.
- 물체탐지
    - 입력 : 이미지한장
    - 입력들어온 한장의 이미지와 def object_detection_prompt()함수의 아웃풋을 Ovis2 8B 모델에 입력으로 사용한다
    - 출력된 결과를 response로 회신한다.

## 입력 조건
- 입력 이미지는 448x448사이즈의 uint8형태로 요청받는다
- 들어오는 입력 데이터는 bytes형태로 입력받고 http요청을 사용한다

## 엔드포인트
- 요약문 생성 : http://localhost:8666/summary
- 행동인식 : http://localhost:8666/motion
- 물체탐지 : http://localhost:8666/object

## 고려가능한 기술(선택)
- triton inference server
- VLLM
- tensorRT
- transformers
- pytorch
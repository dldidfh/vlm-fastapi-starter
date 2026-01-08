# 구성 방법 추천안

## 목표
- Ovis2 8B 단일 모델로 요약문 생성/행동인식/물체탐지를 효율적으로 서빙한다.
- 입력은 448x448 uint8 이미지 bytes, HTTP로 수신한다.

## 전체 구조
- API 서버 (HTTP)
  - 엔드포인트: `/summary`, `/motion`, `/object`
  - 요청 파싱, 검증, 전처리, 후처리 담당
- 추론 엔진
  - Ovis2 8B 로딩 및 배치 추론
  - 프롬프트 생성 모듈과 통합
- 공통 유틸
  - 이미지 디코딩/리사이즈
  - bytes -> ndarray 변환
  - 로깅/모니터링

## 디렉토리/모듈 제안
- `app/`
  - `main.py` : 서버 시작점
  - `routes.py` : 라우터 정의
  - `schemas.py` : 요청/응답 스키마
- `core/`
  - `model.py` : Ovis2 8B 로더, 추론 함수
  - `prompts.py` : `category_prompt`, `summarize_prompt`, `motion_recognition_prompt`, `object_detection_prompt`
  - `pipeline.py` : 서비스별 파이프라인 로직
- `utils/`
  - `image.py` : bytes 디코딩, 448x448 검증
  - `postprocess.py` : 결과 파싱

## 서비스별 처리 흐름
- 요약문 생성 `/summary`
  1) 연속 이미지 입력 검증
  2) `category_prompt()` 생성
  3) 카테고리 분류 추론
  4) 출력 파싱 후 `summarize_prompt()`에 주입
  5) 요약문 생성 추론
  6) 결과 응답
- 행동인식 `/motion`
  1) 연속 이미지 입력 검증
  2) `motion_recognition_prompt()` 생성
  3) 행동인식 추론
  4) 결과 응답
- 물체탐지 `/object`
  1) 단일 이미지 입력 검증
  2) `object_detection_prompt()` 생성
  3) 물체탐지 추론
  4) 결과 응답

## 기술 선택 가이드
- 단일 서버/간단한 배포: `transformers` + `torch`로 직접 서빙
- 고성능/배치 처리: `vLLM` 고려
- 멀티 모델/다중 GPU 관리: `Triton Inference Server` 고려
- 지연 최소화 최우선: `TensorRT` 경량화 고려

## 성능/안정성 체크리스트
- 요청당 최대 이미지 수 제한
- 배치 추론으로 GPU 활용 극대화
- 타임아웃/재시도 정책
- 프롬프트/응답 로깅
- 모델 워밍업 및 헬스체크

## 테스트 권장
- 입력 데이터 유효성 테스트
- 엔드포인트별 응답 형식 테스트
- 배치/동시성 부하 테스트

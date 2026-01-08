# Ovis2 + Qwen3 VLM Serving

## 목적
- Ovis2 8B 모델로 요약문 생성/행동인식/물체탐지를 서빙한다.
- 카테고리 분류는 qwen3-vl-8b-instruct로 수행한다.

## 모델 구성
- 요약문 생성/행동인식/물체탐지: Ovis2 8B
- 카테고리 분류: qwen3-vl-8b-instruct

## 서비스별 처리 흐름
- 요약문 생성 `/summary`
  1) 연속 이미지 입력 검증 (448x448 RGB)
  2) qwen3-vl-8b-instruct로 카테고리 분류
  3) 카테고리를 summarize 프롬프트에 주입
  4) Ovis2 8B로 요약문 생성
- 행동인식 `/motion`
  1) 연속 이미지 입력 검증 (448x448 RGB)
  2) Ovis2 8B로 행동인식 생성
- 물체탐지 `/object`
  1) 단일 이미지 입력 검증 (448x448 RGB)
  2) Ovis2 8B로 물체탐지 생성

## 입력 조건
- 이미지: 448x448, uint8, RGB
- 요청 형식: `multipart/form-data`
  - `/summary`, `/motion` : 다중 파일 업로드
  - `/object` : 단일 파일 업로드

## 엔드포인트
- 요약문 생성: `http://localhost:8666/summary`
- 행동인식: `http://localhost:8666/motion`
- 물체탐지: `http://localhost:8666/object`

## 설치 및 실행
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## 환경 변수
- `OVIS_MODEL_ID` (default: `AIDC-AI/Ovis2-8B`)
- `OVIS_DEVICE` (default: `auto`)
- `OVIS_DTYPE` (default: `bfloat16`)
- `OVIS_MULTIMODAL_MAX_LENGTH` (default: `32768`)
- `QWEN_MODEL_ID` (default: `qwen3-vl-8b-instruct`)
- `QWEN_DEVICE` (default: `auto`)
- `QWEN_DTYPE` (default: `auto`)
- `QWEN_MAX_NEW_TOKENS` (default: `128`)

## 요청 예시
```bash
curl -X POST http://localhost:8666/summary \
  -F "files=@/path/to/img1.png" \
  -F "files=@/path/to/img2.png"

curl -X POST http://localhost:8666/motion \
  -F "files=@/path/to/img1.png" \
  -F "files=@/path/to/img2.png"

curl -X POST http://localhost:8666/object \
  -F "file=@/path/to/img.png"
```

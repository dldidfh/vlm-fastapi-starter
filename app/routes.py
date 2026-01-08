from typing import List

from fastapi import APIRouter, File, UploadFile

from app.schemas import ObjectResponse, SummaryResponse, MotionResponse
from core.model import get_models
from core.pipeline import run_motion, run_object, run_summary
from utils.image import decode_and_validate

router = APIRouter()


@router.post("/summary", response_model=SummaryResponse)
async def summary(files: List[UploadFile] = File(...)) -> SummaryResponse:
    images = []
    for file in files:
        data = await file.read()
        images.append(decode_and_validate(data))
    ovis_model, qwen_model = get_models()
    result = run_summary(ovis_model, qwen_model, images)
    return SummaryResponse(result=result)


@router.post("/motion", response_model=MotionResponse)
async def motion(files: List[UploadFile] = File(...)) -> MotionResponse:
    images = []
    for file in files:
        data = await file.read()
        images.append(decode_and_validate(data))
    ovis_model, _ = get_models()
    result = run_motion(ovis_model, images)
    return MotionResponse(result=result)


@router.post("/object", response_model=ObjectResponse)
async def object_detect(file: UploadFile = File(...)) -> ObjectResponse:
    data = await file.read()
    image = decode_and_validate(data)
    ovis_model, _ = get_models()
    result = run_object(ovis_model, image)
    return ObjectResponse(result=result)

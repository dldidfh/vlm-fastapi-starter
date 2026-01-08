from pydantic import BaseModel


class SummaryResponse(BaseModel):
    result: str


class MotionResponse(BaseModel):
    result: str


class ObjectResponse(BaseModel):
    result: str

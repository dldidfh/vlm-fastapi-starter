from io import BytesIO

from PIL import Image


def decode_and_validate(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")
    if image.size != (448, 448):
        raise ValueError("Image must be 448x448")
    return image

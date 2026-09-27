import base64
import io

from PIL import Image


def decode_upload(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes))


def encode_image_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
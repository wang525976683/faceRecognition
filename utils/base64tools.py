import base64
from PIL import Image
from io import BytesIO


def img2base64(image):
    """图片格式转base64"""
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_string


def imgFile2base64(image_path):
    """
    给定图片路径转换为base64
    :param image_path: 图片路径
    :return:
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def base642imgFile(base64_str, output_path):
    """
    base64格式转化为图片，并且保存
    :param base64_str:
    :param output_path:
    :return:
    """
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(output_path)


def base642img(base64_str):
    """
    base64格式转化为图片格式
    :param base64_str:
    :return:
    """
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img

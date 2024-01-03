import os
from os.path import join
from PIL import Image


def capture_image_area(src_photo, boxes, dst_photo_path=None, is_linux=False):
    """
    图像中截取指定矩形区域
    :param src_photo: 原图
    :param boxes: 框的对角坐标，格式为(left, upper, right, lower）
    :param dst_photo_path: 截取后图片保存位置，为None则不保存
    :param is_linux: 是否为linux的文件系统
    :return: 截取后的图片
    """
    assert len(boxes) == 4, "需要指定截取框的对角坐标，格式为(left, upper, right, lower）"
    from PIL import Image
    # 打开图片
    img = Image.open(src_photo)
    # 定义截取的区域 (left, upper, right, lower)
    # 使用crop()函数截取图片
    img_crop = img.crop(boxes)
    if dst_photo_path is not None:
        if is_linux:
            dst_photo_name = src_photo.split('/')[-1]
        else:
            dst_photo_name = src_photo.split('\\')[-1]
        # 保存截取后的图片
        if not os.path.exists(dst_photo_path):
            os.mkdir(dst_photo_path)
        img_crop.save(join(dst_photo_path, dst_photo_name))
    return img_crop


def resize_image(img, width=128, height=172, output_image_path=None):
    """
    伸缩图片
    :param img:输入图片如果是str类型，则认为是图片路径
    :param width:修改后的宽度
    :param height:修改后的高度
    :param output_image_path:如果需要保存在指定路径
    :return: resized_image，修改后的图片
    """
    from PIL import Image
    if isinstance(img, str):
        img = Image.open(img)
    resized_image = img.resize((width, height), Image.LANCZOS)
    if output_image_path is not None:
        resized_image.save(output_image_path)
    return resized_image


def save_img(img, name_dir, dst_photo_name, save_path=None, is_linux=False):
    if save_path is None:
        save_path = os.getcwd()

    if name_dir is None:
        name_dir = "unknown"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(join(save_path, name_dir)):
        os.mkdir(join(save_path, name_dir))
    save_face_path = join(save_path, name_dir, dst_photo_name)
    try:
        # 保存截取后的图片
        img.save(save_face_path)
        print("INFO : Save img to %s" % save_face_path)
    except Exception as e:
        print("ERROR : Save img to %s error ,message:%s" % (save_face_path, e))


def resize_and_add_border(image, border_color="white", expected_width=128, expected_height=172):
    """
    将图片缩放，并且限定为固定大小，无法靠缩放填充的部分，用空白填充
    :param image: 待处理图片
    :param border_color:填充颜色
    :param expected_width:预计最终的宽度
    :param expected_height:预计最终的高度
    :return:缩放填充后的图片
    """
    if isinstance(image, str):
        # 打开图片
        img = Image.open(image)
    else:
        img = image
    # 获取图片大小
    width, height = img.size
    telescopic_coefficient_width = float(width) / expected_width
    telescopic_coefficient_height = float(height) / expected_height
    if telescopic_coefficient_width > telescopic_coefficient_height:
        img = resize_image(img, int(width / telescopic_coefficient_width), int(height / telescopic_coefficient_width))
    else:
        img = resize_image(img, int(width / telescopic_coefficient_height), int(height / telescopic_coefficient_height))
    r_width, r_height = img.size
    # 创建一个新的空白图片，大小与原图一致，背景颜色为边框颜色
    new_img = Image.new('RGB', (expected_width, expected_height), border_color)
    # 将原图复制到新的图片中，位置为上下左右边框之外的部分
    left_border = int((expected_width - r_width) / 2)
    top_border = int((expected_height - r_height) / 2)
    new_img.paste(img, (left_border, top_border))
    return new_img
import matplotlib.pyplot as plt
from matplotlib import rcParams
from modelscope.pipelines import pipeline
import os
from os.path import join
from tqdm import tqdm
from modelscope.outputs import OutputKeys

import numpy as np

rcParams['font.family'] = 'SimHei'

plt.rcParams['font.family'] = 'serif'  # 使用serif作为默认字体，这个字体一般支持数学符号

photo_path = r"example\userimg"


def get_faces_data(face_path, max_example=100):
    """
    :param face_path: 脸文件夹
    :param max_example: 每个人最多的脸部样例图片
    :return: list[(name,list[face_path])]
    """
    names = [item for item in os.listdir(face_path) if os.path.isdir(join(face_path, item))]
    faces_list = []
    for name in names:
        faces = [join(face_path, name, face)  # 拼接完整路径
                 for face in os.listdir(join(face_path, name))  # 遍历每个名字下的所有面部照片
                 if face.endswith('.jpg') or face.endswith('.png')]  # 过滤掉不是图片格式的文件
        faces_list.append((name, faces[-max_example:]))
    return faces_list


def img2vector(photo_path, inference=None):
    if inference is None:
        inference = pipeline("face_recognition", model='bubbliiiing/cv_retinafce_recognition', model_revision='v1.0.3')

    faces_list = get_faces_data(photo_path)
    face2index = {}
    indices = []
    vectors = []
    for name, faces in faces_list:
        if name not in face2index:
            face2index[name] = len(face2index) + 1

        tq = tqdm(range(len(faces)))
        for i in tq:
            face = faces[i]
            try:
                emb = inference(dict(user=face))[OutputKeys.IMG_EMBEDDING]
                if len(emb) >= 1 and name in face2index:
                    indices.append(face2index[name])
                    vectors.append(emb[0])
                tq.set_description("img2vector User:%s Step %d" % (name, i))
            except Exception as e:
                print(f"WARN : face {face} 解析失败 message :[{e}]")

    vectors = np.array(vectors)

    return face2index, indices, vectors


face2index, indices, vectors = img2vector(photo_path)
show = vectors

# 计算每一列的平均值
mean_values = np.mean(vectors, axis=0)

# 计算每一列的均方误差
mse = np.mean((vectors - mean_values) ** 2, axis=0)

sorted_indices = np.argsort(mse, kind='mergesort')

# 创建数据
x = show[:, sorted_indices[-1]]
y = show[:, sorted_indices[-2]]
z = show[:, sorted_indices[-3]]
colors = [i * 100 for i in indices]

sizes = 10

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=1.0, )

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('颜色')

# 设置坐标轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 显示图形
plt.show()

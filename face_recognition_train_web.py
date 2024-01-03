import gradio as gr
import pandas as pd

from FaceRecognition2.face_recognition import *

user_state_dict = dict(unique_identification_id=[], name=[], folder_input=[])
css_style = """  
<style>  
table {  
    width: 100%;  
}  
</style>  
"""

face_recognition_dict = {}


def userInformationEntry(folder_input, name, unique_identification_id):
    """
    单个添加训练用户
    :param folder_input: 用户对应的照片目录
    :param name:姓名
    :param unique_identification_id:唯一标识
    :return:
    """
    if not (folder_input and name and unique_identification_id):
        raise gr.Error(
            f"folder_input, name and unique_identification_id cannot be empty，your input："
            f"{dict(folder_input=folder_input, name=name, unique_identification_id=unique_identification_id)}")
    if unique_identification_id in user_state_dict["unique_identification_id"]:
        index = user_state_dict["unique_identification_id"].index(unique_identification_id.strip())
        user_state_dict["name"][index] = name.strip()
        user_state_dict["folder_input"][index] = folder_input.strip()
    else:
        user_state_dict["unique_identification_id"].append(unique_identification_id.strip())
        user_state_dict["name"].append(name.strip())
        user_state_dict["folder_input"].append(folder_input.strip())

    return pd.DataFrame(user_state_dict).to_html() + css_style


def deleteByID(id):
    """
    根据id删除已经添加到待训练列表的用户
    :param id: id
    :return:
    """
    if not isinstance(id, int):
        id = int(id)
    user_state_dict["unique_identification_id"].pop(id)
    user_state_dict["name"].pop(id)
    user_state_dict["folder_input"].pop(id)
    return pd.DataFrame(user_state_dict).to_html() + css_style


def readFile(file):
    """
    批量添加用户，目前仅支持csv文件，格式如下：
    unique_identification_id,name,folder_input
    0,username1,folder1
    1,username2,folder2
    2,username3,folder3

    :param file: 文件路径
    :return:
    """
    with open(file.name, 'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            title = lines[0]
        else:
            raise gr.Error('This file is empty. Please prepare the file in the following format:\n'
                           """
                           unique_identification_id,name,folder_input
                           0,username1,folder1
                           1,username2,folder2
                           2,username3,folder3
                           ......
                           """)
        if title.strip() != "unique_identification_id,name,folder_input":
            raise gr.Error('Please use the correct header description :unique_identification_id,name,folder_input')
        for l in lines[1:]:
            split = l.split(',')
            if len(split) == 3:
                unique_identification_id, name, folder_input = split
                if unique_identification_id in user_state_dict["unique_identification_id"]:
                    index = user_state_dict["unique_identification_id"].index(unique_identification_id.strip())
                    user_state_dict["name"][index] = name.strip()
                    user_state_dict["folder_input"][index] = folder_input.strip()
                else:
                    user_state_dict["unique_identification_id"].append(unique_identification_id.strip())
                    user_state_dict["name"].append(name.strip())
                    user_state_dict["folder_input"].append(folder_input.strip())

    return pd.DataFrame(user_state_dict).to_html() + css_style


def training(sql_path, faiss_index_file, save_face_path, maximum_sampling, progress=gr.Progress(track_tqdm=True)):
    """
    训练启动方法
    :param sql_path: 训练之后持久化的位置，可以写用mysql或者sqlite，详情可以参照sqlalchemy官网
    :param faiss_index_file: faiss index保存的位置
    :param save_face_path: 截取后的面部文件保存位置
    :param maximum_sampling: 每个用户最多取样的照片数目
    :param progress: 进度条打印（无需在意）
    :return:
    """
    if len(face_recognition_dict.keys()) == 0:
        raise gr.Error('在训练之前,请先创建连接......')
    if len(user_state_dict["unique_identification_id"]) == 0:
        raise gr.Error('Please add users after training......')

    maximum_sampling = int(maximum_sampling)

    args = dict(save_face_path=save_face_path,
                sql_path=sql_path,
                faiss_index_file=faiss_index_file,
                unique_identification=user_state_dict,
                maximum_sampling=maximum_sampling)

    face_recognition_dict[0](args, is_train=True)
    return "训练完成！！"


def createConnection(sql_path, faiss_index_file):
    face_recognition_dict[0] = FaceRecognition(sql_path=sql_path, faiss_index_file=faiss_index_file)
    return "连接状态:已连接！！！"


"""
#### 界面部分
"""
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Column():
                gr.Markdown(f'<p style="font-family: Arial; font-weight: bold; font-size: 18px;">数据库</p>')
                sql_path = gr.Textbox(value="sqlite:///FaceRecognition.db", label='关系型数据库')
                faiss_index_file = gr.Textbox(value=r"faissIndex\FaceRecognition.index", label='Faiss 向量数据库')
                connection_state = gr.Text(label='connection state', value='连接状态:未连接......')
                createConnection_button = gr.Button("创建连接")
            gr.Markdown(f'<p style="font-family: Arial; font-weight: bold; font-size: 18px;">训练导入</p>')
            with gr.Tab("整体导入/修改"):
                file = gr.File(type='file', value='example/example.csv')
                read_button = gr.Button('读取')
            with gr.Tab("单人导入/修改"):
                unique_identification_id = gr.Textbox(label='用户唯一标识')
                name = gr.Textbox(label='姓名')
                folder_input = gr.Textbox(label='照片文件夹')
                add_set_button = gr.Button("添加 / 修改")
            with gr.Tab("参数设置"):
                save_face_path = gr.Textbox(value="saveSubFace", label='面部地址')
                maximum_sampling = gr.Number(value=20, label='每个用户照片最多取样数目')

        with gr.Column():
            gr.Markdown(f'<p style="font-family: Arial; font-weight: bold; font-size: 18px;">用户列表</p>')
            display = gr.HTML(label='已添加用户')
            delete_id = gr.Number(label='需删除id')
            delete_button = gr.Button("删除")

    with gr.Column():
        with gr.Tab("训练"):
            with gr.Row():
                train_button = gr.Button("Train")
        train_progress = gr.Textbox(label='Training progress')

    add_set_button.click(userInformationEntry, inputs=[folder_input, name, unique_identification_id], outputs=display)
    delete_button.click(deleteByID, inputs=[delete_id], outputs=display)
    read_button.click(readFile, inputs=[file], outputs=display)
    train_button.click(training, inputs=[sql_path, faiss_index_file, save_face_path, maximum_sampling],
                       outputs=train_progress)

    createConnection_button.click(createConnection, inputs=[sql_path, faiss_index_file], outputs=[connection_state])

demo.queue()
demo.launch()

import gradio as gr

from FaceRecognition2.face_recognition import *

face_recognition_dict = {}


def createConnection(sql_path, faiss_index_file):
    face_recognition_dict[0] = FaceRecognition(sql_path=sql_path, faiss_index_file=faiss_index_file)
    return "连接状态:已连接！！！"


def prediction(input_img, distance_threshold):
    if len(face_recognition_dict.keys()) == 0:
        raise gr.Error('在预测之前,请先创建连接......')
    args = dict(photo_file_path=input_img,
                distance_threshold=distance_threshold)

    pred = face_recognition_dict[0](args, is_train=False)

    if pred is not None and pred:
        pred = {str(k): dict(unique_identification_id=v['unique_identification_id'],
                             name=v["name"],
                             confidence=f'%{v["score"] * 100.0} ')
                for k, v in
                sorted(pred.items(), key=lambda item: item[1]["score"], reverse=True)}

    return pred


#
with gr.Blocks() as blocks:
    gr.Markdown(f'<p style="font-family: Arial; font-weight: bold; font-size: 18px;">连接设置</p>')
    sql_path = gr.Textbox(value="sqlite:///FaceRecognition.db", label='数据库地址')
    faiss_index_file = gr.Textbox(value=r"faissIndex\FaceRecognition.index", label='faiss索引')
    distance_threshold = gr.Number(value=1.5, label='最大空间距离')
    connection_state = gr.Text(label='connection state', value='连接状态:未连接......')
    createConnection_button = gr.Button("创建连接")
    gr.Markdown(f'<p style="font-family: Arial; font-weight: bold; font-size: 18px;">人脸识别</p>')
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath")
            predict_button = gr.Button('识别')
        with gr.Column():
            predictionTB = gr.JSON(label="预测概览")
    predict_button.click(prediction, inputs=[input_img, distance_threshold],
                         outputs=[predictionTB])
    createConnection_button.click(createConnection, inputs=[sql_path, faiss_index_file], outputs=[connection_state])

blocks.launch()

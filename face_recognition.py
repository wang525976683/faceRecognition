from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from os.path import join
from tqdm import tqdm
from modelscope.outputs import OutputKeys
import numpy as np
import faiss
import logging
from FaceRecognition2.utils import *
from FaceRecognition2.facedb.sql import *
from typing import Dict, List

logging.basicConfig(level=logging.INFO)


#
class FaceRecognition:
    def __init__(self, inference_model='bubbliiiing/cv_retinafce_recognition',
                 retina_face_detection_model='damo/cv_resnet50_face-detection_retinaface',
                 sql_path="mysql+mysqlconnector://root:root@localhost:3306/escai",
                 faiss_index_file='faissIndex\\FaceRecognition.index'):
        """
        :param inference_model: 面部特征向量提取模型
        :param retina_face_detection_model: 人脸定位模型
        :param sql_path: 数据库持久化路径
        :param faiss_index_file: Faiss index索引路径
        """
        self.sql_path = sql_path
        self.faiss_index_file = faiss_index_file
        self.session = None
        # 面部特征提取模型
        self.inference = pipeline("face_recognition", model=inference_model,
                                  model_revision='v1.0.3')
        # 面部定位模型
        self.retina_face_detection = pipeline(Tasks.face_detection, retina_face_detection_model)
        self.session = self.getSqlConnector(sql_path)
        #
        self.history_max_vector_id = self.getMaxVectorID()
        self.current_vector_id = self.history_max_vector_id + 1
        self.delete_face_vector_id = []

    def getFacesDetails(self, users, max_example=20):
        """
        为每张图片丰富信息
        :param users: 人物信息列表
        :param max_example: 每个人最多的脸部样例图片
        :return: list[dict(unique_identification_id=?,name=?,vector_id=?,photo_file_path=?)]
            unique_identification_id:用户唯一索引
            name:用户姓名
            vector_id:面部向量索引id
            photo_file_path:图片完整路径
        """
        details = []

        for user in users:
            unique_identification_id = user.unique_identification_id
            name = user.name
            folder_input = user.folder_input

            img_list = [img for img in os.listdir(folder_input) if img.endswith('.jpg') or img.endswith('.png')]
            for img in img_list[-max_example:]:
                details.append(dict(unique_identification_id=unique_identification_id,
                                    name=name,
                                    vector_id=self.current_vector_id,
                                    photo_file_path=join(folder_input, img)))
                self.current_vector_id = self.current_vector_id + 1
        return details

    def faceLocation2(self, details, retina_face_detection=None):
        """
        传入拍摄的照片，需要先定位人脸，截取，并且调整大小
        :param retina_face_detection: 人脸定位模型
        :return: 返回无法定位到人脸的图片
        """

        if retina_face_detection is None:
            retina_face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
        unable_to_recognize_faces = []  # 无法识别人脸信息
        tq = tqdm(range(len(details)), desc="Face capture and scaling......")
        for i, d in zip(tq, details):
            name = d['name']
            photo_file_path = d['photo_file_path']
            raw_result = retina_face_detection(photo_file_path)
            if raw_result["boxes"]:
                # 截取面部
                img = capture_image_area(photo_file_path, raw_result["boxes"][0])
                # 在不扭曲图像的前提下调整至统一大小，缩放之后用边框调整
                img = resize_and_add_border(img)
                details[i]['sub_face'] = img
            else:
                logging.warning('无法识别面部 : %s' % photo_file_path)
                unable_to_recognize_faces.append(i)
            tq.set_description("Face location User:%s Step %d" % (name, i))
        return unable_to_recognize_faces

    def img2vector(self, details):
        """
        将details中的所有面部信息通过模型
        :return:无法正确将面部信息转化为面部特征向量的图片列表
        """

        if self.inference is None:
            self.inference = inference = pipeline("face_recognition", model='bubbliiiing/cv_retinafce_recognition',
                                                  model_revision='v1.0.3')
        err_face_indices = []
        tq = tqdm(range(len(details)), desc="Facial features converted to vectors......")
        for i, d in zip(tq, details):
            try:
                sub_face_img = d['sub_face']
                emb = self.inference(dict(user=sub_face_img))[OutputKeys.IMG_EMBEDDING]
                details[i]['vector'] = emb[0]
            except Exception as e:
                err_face_indices.append(i)
                logging.warning(f"face {details[i]['photo_file_path']} 解析失败 message :[{e}]")
            tq.set_description("img2vector User progress: %d" % i)
        return err_face_indices

    @staticmethod
    def createFaissIndex(vectors, ids, dim=512, faiss_index_file=r"faissIndex/FaceRecognition.index"):
        """
        将人脸转化的向量保存为添加到faiss库，并持久化
        :param vectors: 图片转化的向量
        :param dim: 向量的维度，这里为512
        :param faiss_index_file: faiss index的路径，路径不能携带中文
        :return:
        """
        try:
            logging.info(f"开始创建人脸索引库：{faiss_index_file}....")
            import faiss
            faissIndex = faiss.IndexFlatL2(dim)
            faissIndex2map = faiss.IndexIDMap(faissIndex)
            faissIndex2map.add_with_ids(vectors, ids)
            faiss.write_index(faissIndex2map, faiss_index_file)  # 此处只能用相对路径 join(index_path, 'face.index'))
            logging.info(f"创建人脸索引库完成，保存在：{faiss_index_file}....")
        except Exception as e:
            logging.error("创建Faiss索引失败，错误：%s" % e)

    def updateFaissIndex(self, vectors, ids, faiss_index_file=r"faissIndex/FaceRecognition.index"):
        """
        将人脸转化的向量保存为添加到faiss库，并持久化
        :param vectors: 图片转化的向量
        :param index_path: faiss index的路径，路径不能携带中文
        :return:
        """
        try:
            logging.info(f"开始更新人脸索引库：{faiss_index_file}....")
            import faiss
            faissIndex2map = self.getFaissIndex(faiss_index_file)
            faissIndex2map.remove_ids(np.array(self.delete_face_vector_id))
            faissIndex2map.add_with_ids(vectors, ids)
            faiss.write_index(faissIndex2map, faiss_index_file)  # 此处只能用相对路径 join(index_path, 'face.index'))
            logging.info(f"更新人脸索引库完成，保存在：{faiss_index_file}....")
        except Exception as e:
            logging.error("更新Faiss索引失败，错误：%s" % e)

    @staticmethod
    def getFaissIndex(faiss_index_file=r"face.index"):
        return faiss.read_index(faiss_index_file)

    def saveFaceImgs(self, details, save_face_path, is_linux=False):
        """
        :param details: 图片列表
        :param save_face_path: 保存的根文件夹
        :param is_linux: 是否为linux系统
        :return: None
        """
        if not os.path.exists(save_face_path):
            os.mkdir(save_face_path)
        tq = tqdm(range(len(details)), desc=f"Save captured facial information to :{save_face_path}")
        for i, d in zip(tq, details):
            unique_identification_id = d["unique_identification_id"]
            name = d["name"]
            photo_file_path = d["photo_file_path"]
            img = d['sub_face']
            save_photo_face_dir = join(save_face_path, f"{unique_identification_id}_{name}")
            if not os.path.exists(save_photo_face_dir):
                os.mkdir(save_photo_face_dir)
            if is_linux:
                dst_face_name = photo_file_path.split('/')[-1]
            else:
                dst_face_name = photo_file_path.split('\\')[-1]
            face_path = join(save_photo_face_dir, dst_face_name)
            img.save(face_path)
            details[i]['sub_face_file'] = face_path
            tq.set_description("Save face img %s Step %d" % (name, i))
        logging.info('面部特征图片保存于 ：%s' % save_face_path)

    @staticmethod
    def listFilter(list, fliter_indices):
        """
        过滤列表元素
        :param list: 待过滤列表
        :param fliter_indices: 需要清理的id
        :return:
        """
        return [value for i, value in enumerate(list) if
                i not in fliter_indices]

    def addDetails2DB(self, index2users: Dict[str, User], face_details: List):
        """
        数据入库
        """
        logging.info(f"开始添加用户以及对应面部信息到数据库:{self.sql_path}")
        try:
            for user in index2users.values():
                self.session.add(user)
            for d in face_details:
                unique_identification_id = d['unique_identification_id']
                vector_id = d['vector_id']
                photo_file_path = d['photo_file_path']
                sub_face_file = d['sub_face_file']
                user = index2users[unique_identification_id]
                photo = Photo(photo_path=photo_file_path, user=user)
                self.session.add(photo)
                face = Face(face_file=sub_face_file, vector_id=vector_id, user=user,
                            photo=photo)
                self.session.add(face)
            # 提交事务，将数据保存到数据库中
            self.session.commit()
            # 关闭会话
            self.session.close()
        except Exception as e:
            logging.error('同步到数据库:%s 出现异常！！异常信息:%s' % (self.sql_path, e))
            self.session.rollback()

    def deleteDetailsByUsers(self, index2users):
        """
        根据用户的唯一标识unique_identification_id，删除该用户下的所有数据，包括用户，照片以及面部信息
        :param index2users:
        :return:
        """
        logging.info(f"清理历史用户以及对应面部信息到数据库:{self.sql_path}")

        try:
            for user in index2users.values():
                users_ = self.session.query(User).filter(
                    User.unique_identification_id == user.unique_identification_id)
                faces = self.session.query(Face).filter(Face.slave_user_id == users_.one().id).all()
                photos = self.session.query(Photo).filter(Photo.slave_user_id == users_.one().id).all()
                for face in faces:
                    self.delete_face_vector_id.append(face.vector_id)
                    self.session.delete(face)
                for photo in photos:
                    self.session.delete(photo)
                for user_ in users_:
                    self.session.delete(user_)
            # 提交事务，将数据保存到数据库中
            self.session.commit()
            # 关闭会话
            self.session.close()
            return True

        except Exception as e:
            logging.error('清理数据库:%s 出现异常！！异常信息:%s' % (self.sql_path, e))
            self.session.rollback()
            return False

    def getMaxVectorID(self):
        """
        获取Face表中的最大vector_id
        :return:
        """
        try:
            result = self.session.query(func.max(Face.vector_id)).scalar()
            if result:
                return result
            else:
                return 0
        except Exception as e:
            return 0

    def faceFormatting(self, users, save_face_path, maximum_sampling):
        """
        整理图片信息
        :param users: 用户列表
        :param save_face_path: 面部图片保存位置
        :return:
            List[dict(
            unique_identification_id：用户唯一标识,
            name：用户姓名,
            vector_id：面部特征向量的唯一id标识，对应 Faiss id,
            photo_file_path：原照片对应路径
            sub_face：截取面部图片
            vector ： 面部特征向量
            )]
        """
        # 获取图片地址
        details = self.getFacesDetails(users, max_example=maximum_sampling)
        # 定位面部并且缩放统一大小
        unable_to_recognize_faces = self.faceLocation2(details)
        # 清理无法识别面部的图片
        details = self.listFilter(details, unable_to_recognize_faces)
        # 保存截取缩放后的图片
        self.saveFaceImgs(details, save_face_path)
        # 转化为面部特征向量
        err_face_indices = self.img2vector(details)
        # 删除格式有误的图片
        details = self.listFilter(details, err_face_indices)

        return details

    def train(self, args):
        """
        :param args:
        :return:
        """
        update_index2users = {}
        add_index2users = {}
        # 用户当前添加用户信息
        pre_add_users = args['unique_identification']

        pre_add_users_ids = pre_add_users['unique_identification_id']
        # 当前用户信息与历史库中的交集
        sqldb_existed_users = self.session.query(User) \
            .filter(User.unique_identification_id.in_(pre_add_users_ids))
        sqldb_existed_user_dict = {}
        # 整理用户并且整理为字典形式
        for u in sqldb_existed_users:
            sqldb_existed_user_dict[u.unique_identification_id] = u

        sqldb_users_unique_identification_ids = [u.unique_identification_id for u in sqldb_existed_users]
        # 划分需要更新的用户和直接添加的用户
        for unique_identification_id, name, folder_input in zip(*pre_add_users.values()):
            if unique_identification_id in sqldb_users_unique_identification_ids:
                update_index2users[unique_identification_id] = User(
                    id=sqldb_existed_user_dict[unique_identification_id].id,
                    unique_identification_id=unique_identification_id,
                    name=name,
                    folder_input=folder_input,
                    faiss_index_file=self.faiss_index_file,
                    created_at=sqldb_existed_user_dict[unique_identification_id].created_at
                )
            else:
                add_index2users[unique_identification_id] = User(
                    unique_identification_id=unique_identification_id,
                    name=name,
                    folder_input=folder_input,
                    faiss_index_file=self.faiss_index_file)

        # 分别整理详细信息
        sql_db_face_details_add = self.faceFormatting(add_index2users.values(), args['save_face_path'],
                                                      args['maximum_sampling'])
        sql_db_face_details_update = self.faceFormatting(update_index2users.values(), args['save_face_path'],
                                                         args['maximum_sampling'])
        # 创建数据库
        self.addDetails2DB(add_index2users, sql_db_face_details_add)

        if self.deleteDetailsByUsers(update_index2users):
            self.addDetails2DB(update_index2users, sql_db_face_details_update)

        vectors = np.array([d['vector'] for d in sql_db_face_details_add + sql_db_face_details_update])
        ids = np.array([d['vector_id'] for d in sql_db_face_details_add + sql_db_face_details_update])
        # 创建Faiss index
        if not os.path.exists(self.faiss_index_file):
            self.createFaissIndex(vectors, ids, faiss_index_file=self.faiss_index_file)
        else:
            self.updateFaissIndex(vectors, ids, faiss_index_file=self.faiss_index_file)

    def predictionImg2Vector(self, photo_file_path):
        if self.retina_face_detection is None:
            self.retina_face_detection = pipeline(Tasks.face_detection,
                                                  'damo/cv_resnet50_face-detection_retinaface')
        raw_result = self.retina_face_detection(photo_file_path)
        if raw_result["boxes"]:
            # 截取面部
            img = capture_image_area(photo_file_path, raw_result["boxes"][0])
            # 调整至统一大小
            img = resize_image(img)
        else:
            logging.warning('无法识别面部 : %s' % photo_file_path)
            return None

        if self.inference is None:
            self.inference = pipeline("face_recognition", model='bubbliiiing/cv_retinafce_recognition',
                                      model_revision='v1.0.3')
        try:
            emb = self.inference(dict(user=img))[OutputKeys.IMG_EMBEDDING]
            vector = emb[0]
        except Exception as e:
            logging.error(f'无法将面部信息转换为向量,报错信息:{e}')
            return None
        return vector

    def getSqlConnector(self, sql_path):
        # 创建会话
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(sql_path)
        # 创建表
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        return session

    def searchSimilarity(self, vector, top):
        """
        1.检索Faiss数据库返回和向量vector距离最近的top条数据的vector_id
        2.通过vector_id去获取sqldb中的对应的图片以及用户信息
        :param vector:待检索向量
        :param top:最多返回的向量个数
        :return:List[dict(unique_identification_id:用户唯一标识,
                                      name:名字,
                                      folder_input:照片目录,
                                      slave_user_id:用户对应的主键id,
                                      slave_photo_id:源图片id,
                                      sub_face_file:截取的面部图片id,
                                      photo_path:照片完整路径,
                                      vector_id:faiss向量索引id,
                                      distance:空间距离]
        """
        faiss_index = self.getFaissIndex(faiss_index_file=self.faiss_index_file)
        vector = np.array(vector).reshape([-1, 512])
        # D:distance I:index
        D, I = faiss_index.search(vector, top)
        I = I.reshape(-1).tolist()
        D = D.reshape(-1).tolist()

        index2distance = {i: d for i, d in zip(I, D)}

        result = self.session.query(User, Photo, Face).filter(
            and_(
                Face.vector_id.in_(I),
                Face.slave_photo_id == Photo.id,
                Face.slave_user_id == User.id
            )).all()
        result_format = []

        for user, photo, face in result:
            result_format.append(dict(unique_identification_id=user.unique_identification_id,
                                      name=user.name,
                                      folder_input=user.folder_input,
                                      slave_user_id=face.slave_user_id,
                                      slave_photo_id=face.slave_photo_id,
                                      sub_face_file=face.face_file,
                                      photo_path=photo.photo_path,
                                      vector_id=face.vector_id,
                                      distance=index2distance[face.vector_id]
                                      ))
        return result_format

    def assessment(self, result_format, distance_threshold):
        """
        对预测结果进行评估计算，


        :param result_format:历史数据库中，Top N条和待预测照片最匹配的照片详细信息。
        :param distance_threshold:空间距离阀值
        :return:dict(unique_identification_id:dict(unique_identification_id:用户唯一索引,
                                                      name:姓名,
                                                      sub_face_files:面部的图像列表,
                                                      score:评分,
                                                      distances:空间距离列表))
        """

        pred = {}

        for d in result_format:
            if d['distance'] >= distance_threshold:
                continue
            unique_identification_id = d['unique_identification_id']
            name = d['name']
            sub_face_file = d['sub_face_file']
            if unique_identification_id in pred.keys():
                sub_face_files = pred[unique_identification_id]["sub_face_files"]
                sub_face_files.append(sub_face_file)
                face_sum = len(sub_face_files)
                pred[unique_identification_id]['score'] = float(face_sum) / len(result_format)
                pred[unique_identification_id]['distances'].append(d['distance'])
            else:
                pred[unique_identification_id] = dict(unique_identification_id=unique_identification_id, name=name,
                                                      sub_face_files=[sub_face_file],
                                                      score=1.0 / len(result_format),
                                                      distances=[d['distance']])
        return pred

    def prediction(self, photo_file_path, top=10, distance_threshold=1.5):
        """
        :param photo_file_path: 待预测图
        :param inference: 面部特征提取模型
        :param top: 匹配近似的几张图片
        :return:
        """
        # 图片转向量
        vector = self.predictionImg2Vector(photo_file_path)
        if vector is None:
            return None
        # 匹配相似性
        result_format = self.searchSimilarity(vector, top)

        pred = self.assessment(result_format, distance_threshold)
        return pred

    def __call__(self, args, is_train=True):
        """
        :param args:
        :param is_train:如果为True则为训练模型，False为预测模式
        :return:
        """
        if is_train:
            return self.train(args)
        else:
            return self.prediction(args["photo_file_path"], distance_threshold=args["distance_threshold"])


if __name__ == '__main__':
    # sql_path = r'sqlite:///FaceRecognition.db'
    sql_path = r'mysql+mysqlconnector://root:root@localhost:3306/escai'
    faiss_index_file = r'faissIndex\\FaceRecognition.index'
    args = {
        'save_face_path': 'saveSubFace',
        'unique_identification':
            {'unique_identification_id': ['0', '1', '2', '3', '4', '5'],
             'name': ['dilireba', 'jiangwen', 'lipeiyu', 'pengyuyan', 'zhangziyi', 'zhaoliying'],
             'folder_input': ['example/userimg/dilireba', 'example/userimg/jiangwen', 'example/userimg/lipeiyu',
                              'example/userimg/pengyuyan', 'example/userimg/zhangziyi',
                              'example/userimg/zhaoliying']},
        'maximum_sampling': 20,
        'photo_file_path': r"example\userimg\dilireba\15911604352.jpg",
        "distance_threshold": 1.5}

    fr = FaceRecognition(sql_path=sql_path, faiss_index_file=faiss_index_file)
    # 训练测试
    fr(args=args, is_train=True)
    # 预测测试
    pre = fr(args=args, is_train=True)

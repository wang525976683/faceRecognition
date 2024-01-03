import os

os.environ["SQLALCHEMY_SILENCE_UBER_WARNING"] = "1"

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKeyConstraint,
    String,
    Text,
    and_,
    create_engine,
    func,
    text,
    Integer,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()  # type: Any


class ORMBase(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, default=None)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), server_onupdate=func.now())


class User(ORMBase):
    """
    保存个人信息
    """
    __tablename__ = "user"

    name = Column(String(100), nullable=False, comment='User name')
    unique_identification_id = Column(String(100), nullable=False,
                                      comment="Each person's unique identification.")
    faiss_index_file = Column(String(100), comment='Faiss index file')
    folder_input = Column(String(100), comment='User photo dir')

    face = relationship('Face', back_populates='user')
    photo = relationship('Photo', back_populates='user')


class Photo(ORMBase):
    """
    保存原图
    """
    __tablename__ = 'photo'
    photo_path = Column(Text, nullable=True, comment='path of original.')
    slave_user_id = Column(Integer, ForeignKey('user.id'),
                           comment='Associate the id of people, each person can have multiple facial ids.')
    face = relationship('Face', back_populates='photo')
    user = relationship('User', back_populates='photo')


class Face(ORMBase):
    """
    保存完整图片截取后的面部图片转化为basce64后的格式
    """
    __tablename__ = 'face'

    face_file = Column(Text, nullable=True, comment='Path of facial pictures')
    slave_user_id = Column(Integer, ForeignKey('user.id'),
                           comment='Associate the id of people, each person can have multiple facial ids.')
    slave_photo_id = Column(Integer, ForeignKey('photo.id'), comment="Source of facial information.")

    # Faiss index，对应图片的唯一索引
    vector_id = Column(Integer(), nullable=True, comment='Faiss index number')
    user = relationship('User', back_populates='face')
    photo = relationship('Photo', back_populates='face')

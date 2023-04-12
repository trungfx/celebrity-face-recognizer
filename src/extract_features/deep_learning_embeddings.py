from deepface import DeepFace
from tqdm import tqdm
import os
import re
import numpy as np


def embedding_vector_with_detect(image_path, model="VGG-Face", normalization="base", detector_backend="mtcnn"):
    if model not in ["Facenet", "Facenet512", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "Dlib", "SFace", "ArcFace"]:
        raise ValueError("Invalid model name")

    embeddings = []
    facial_area = []
    embedding_objs = DeepFace.represent(img_path=image_path,
                                        model_name=model,
                                        detector_backend=detector_backend,
                                        normalization=normalization,
                                        enforce_detection=False,
                                        align=True)

    for embedding_obj in embedding_objs:
        embeddings.append(embedding_obj["embedding"])
        facial_area.append(embedding_obj["facial_area"])

    return embeddings, facial_area


def embedding_vector(image_path, model="VGG-Face", normalization="base"):
    if model not in ["Facenet", "Facenet512", "VGG-Face", "OpenFace", "DeepFace", "DeepID", "Dlib", "SFace", "ArcFace"]:
        raise ValueError("Invalid model name")

    embedding_objs = DeepFace.represent(img_path=image_path,
                                        model_name=model,
                                        detector_backend="skip",
                                        normalization=normalization,
                                        enforce_detection=False,
                                        align=True)
    return embedding_objs[0]["embedding"]


def number_label(folder_name):
    match = re.search(r'\d+', folder_name)  # Tìm kiếm một chuỗi các số liên tiếp
    if match:
        number = int(match.group())  # Chuyển chuỗi số thành số nguyên
        return number
    return 0


def save_embeddings(face_path, save_path, file_name, model="VGG-Face", normalization="base"):
    data = []
    for folder in tqdm(os.listdir(face_path), position=0):
        label = number_label(folder)
        if label > 0:
            folder_path = os.path.join(face_path, folder)
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)

                embedding = embedding_vector(image_path, model=model, normalization=normalization)
                data.append([embedding, label])

    # Chuyển embedding và label về dạng numpy array
    embedding = np.array([i[0] for i in data])
    label = np.array([i[1] for i in data])

    # Lưu embedding và label vào file numpy
    os.makedirs(save_path, exist_ok=True)
    np_path = os.path.join(save_path, file_name)
    np.savez(np_path, embedding=embedding, label=label)


def save_dl_embeddings_with_id(face_path, save_path, file_name, model="VGG-Face", normalization="base"):
    data = []
    for folder in tqdm(os.listdir(face_path), position=0):
        folder_path = os.path.join(face_path, folder)
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)

            embedding = embedding_vector(image_path, model=model, normalization=normalization)
            label = folder
            data.append([embedding, label])

    # Chuyển embedding và label về dạng numpy array
    embedding = np.array([i[0] for i in data])
    label = np.array([i[1] for i in data])

    # Lưu embedding và label vào file numpy
    os.makedirs(save_path, exist_ok=True)
    np_path = os.path.join(save_path, file_name)
    np.savez(np_path, embedding=embedding, label=label)

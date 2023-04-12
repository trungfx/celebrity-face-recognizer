from deep_learning_embeddings import embedding_vector_with_detect
import numpy as np
import pickle
import json


def load_model(model_path=None):
    if not model_path:
        model_path = "../data/models/knn_5_cosine_distance_vggface.pkl"

    # Load model
    with open(model_path, 'rb') as f:
        knn_vgg_face_model = pickle.load(f)

    return knn_vgg_face_model


def load_celeb():
    # load data từ json
    with open('../data/json/vietnam_celeb.json', 'r', encoding='utf-8') as f:
        celeb = json.load(f)

    return celeb


def get_data(predictions):
    predicted_celebs = []
    predicted_celeb_ids = set()

    celeb = load_celeb()

    for prediction in predictions:
        if prediction not in predicted_celeb_ids:
            for celeb_info in celeb:
                if celeb_info['id'] == prediction:
                    predicted_celebs.append(celeb_info)
                    predicted_celeb_ids.add(prediction)

    return predicted_celebs


def predict(image_path):
    embeddings, facial_area = embedding_vector_with_detect(image_path,
                                                           model="VGG-Face",
                                                           normalization="base",
                                                           detector_backend="mtcnn")

    embeddings = np.array(embeddings)
    knn = load_model()
    predictions = knn.predict(embeddings)
    result_data = get_data(predictions)

    return result_data

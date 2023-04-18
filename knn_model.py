from deep_learning_embeddings import embedding_vector_with_detect
import numpy as np
import pickle
import json


def load_model(model_path=None):
    if not model_path:
        model_path = "data/models/knn_5_cosine_distance_vggface.pkl"

    # Load model
    with open(model_path, 'rb') as f:
        knn_vgg_face_model = pickle.load(f)

    return knn_vgg_face_model


def load_celeb():
    # load data từ json
    with open('data/json/vietnam_celeb.json', 'r', encoding='utf-8') as f:
        celeb = json.load(f)

    return celeb


def get_data(predictions):
    predicted_celebs = []

    celeb = load_celeb()

    for prediction in predictions:
        for celeb_info in celeb:
            if celeb_info['id'] == prediction:
                predicted_celebs.append(celeb_info)

    return predicted_celebs


def predict(image_path):
    embeddings, facial_area = embedding_vector_with_detect(image_path,
                                                           model="VGG-Face",
                                                           normalization="base",
                                                           detector_backend="mtcnn")

    if len(embeddings) == 0:
        return []

    embeddings = np.array(embeddings)
    knn = load_model()
    # predictions = knn.predict(embeddings)

    # Tính xác xuất
    y_pred_prob = knn.predict_proba(embeddings)
    y_pred_class = knn.classes_[np.argmax(y_pred_prob, axis=1)]
    y_pred_prob_max = np.max(y_pred_prob, axis=1)

    # Lọc với ngưỡng
    threshold = 0.6
    y_pred_threshold = np.where(y_pred_prob_max > threshold, y_pred_class, None)

    predictions = list(filter(None, y_pred_threshold))
    facial_area = [fa for i, fa in enumerate(facial_area) if y_pred_threshold[i] is not None]

    # Lấy dữ liệu người nổi tiếng
    result_data = get_data(predictions)

    # Thêm vị trí khuôn mặt
    if len(result_data) > 0:
        for i in range(len(result_data)):
            result_data[i]['facial_area'] = facial_area[i]

    return result_data

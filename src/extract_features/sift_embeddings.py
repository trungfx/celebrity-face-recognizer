import cv2
import numpy as np
import os
from tqdm import tqdm


def compute_sift_vector(image_path):
    # Đọc ảnh đầu vào
    img = cv2.imread(image_path)

    # Chuyển đổi ảnh màu sang ảnh xám
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Khởi tạo đối tượng trích xuất đặc trưng SIFT với 256 chiều
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=128)

    # Tính toán các điểm đặc trưng và vector đặc trưng SIFT
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Tính toán vector đặc trưng SIFT với 128 chiều
    if descriptors is None:
        descriptors = np.zeros((128, 128))
    elif descriptors.shape[0] < 128:
        descriptors = np.concatenate((descriptors, np.zeros((128 - descriptors.shape[0], 128))))

    # Giới hạn vector đặc trưng chỉ còn 128 chiều
    descriptors = descriptors[:128]

    return descriptors.flatten()


def save_sift_embeddings_with_id(folder_path, embedding_path, file_name):
    # Lấy tên thư mục con làm nhãn
    label = os.listdir(folder_path)

    # Khởi tạo danh sách vector đặc trưng và nhãn tương ứng
    features = []
    labels = []

    # Duyệt qua từng thư mục con
    for i in tqdm(range(len(label)), position=0):
        # Lấy đường dẫn của thư mục con và tất cả các file trong thư mục đó
        sub_folder_path = os.path.join(folder_path, label[i])
        image_paths = [os.path.join(sub_folder_path, f) for f in os.listdir(sub_folder_path)]

        # Tính toán vector đặc trưng và thêm vào danh sách features
        for j in range(len(image_paths)):
            feature = compute_sift_vector(image_paths[j])
            features.append(feature)
            labels.append(label[i])

    # Chuyển embedding và label về dạng numpy array
    embedding = np.array(features)
    label = np.array(labels)

    os.makedirs(embedding_path, exist_ok=True)
    # Chuyển đổi danh sách features và labels sang numpy array và lưu vào file embedding_path
    file_path = os.path.join(embedding_path, file_name)
    np.savez(file_path, embedding=embedding, label=label)

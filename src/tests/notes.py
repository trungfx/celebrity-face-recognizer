import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from src.utils.load_data import load_data

train_path = "../data/embeddings/VGG-Face/train_VGG-Face_embeddings_id.npz"
model_path = "../data/models/"

# Tạo thư mục nếu không tồn tại
os.makedirs(model_path, exist_ok=True)

# Load dữ liệu
x_train, y_train, _, _ = load_data(train_path=train_path)

# Xây dựng mô hình
model = KNeighborsClassifier(metric="cosine", n_neighbors=5)
model.fit(x_train, y_train)

# Đặt tên file theo mô hình, metric và n_neighbors
model_name = "KNN_cosine_5_VGG-Face_id.pkl"

# Lưu mô hình
with open(f"../data/models/{model_name}", 'wb') as f:
    pickle.dump(model, f)

print(f"Mô hình đã được lưu tại /data/models/{model_name}")

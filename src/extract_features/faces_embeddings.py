from src.extract_features.deep_learning_embeddings import save_dl_embeddings_with_id
from src.extract_features.sift_embeddings import save_sift_embeddings_with_id

# Trích xuất đặc trưng với FaceNet
save_dl_embeddings_with_id("../data/faces_train_160", "../data/embeddings/Facenet", "train_Facenet_embeddings_id.npz",
                           model="Facenet", normalization="Facenet")
save_dl_embeddings_with_id("../data/faces_test_160", "../data/embeddings/Facenet", "test_Facenet_embeddings_id.npz",
                           model="Facenet", normalization="Facenet")

# Trích xuất đặc trưng với VGG-Face
save_dl_embeddings_with_id("../data/faces_train_224", "../data/embeddings/VGG-Face", "train_VGG-Face_embeddings_id.npz",
                           model="VGG-Face", normalization="base")
save_dl_embeddings_with_id("../data/faces_test_224", "../data/embeddings/VGG-Face", "test_VGG-Face_embeddings_id.npz",
                           model="VGG-Face", normalization="base")

# Trích xuất đặc trưng với SIFT
save_sift_embeddings_with_id("../data/faces_train_224", "../data/embeddings/SIFT", "train_SIFT_embeddings_id.npz")
save_sift_embeddings_with_id("../data/faces_test_224", "../data/embeddings/SIFT", "test_SIFT_embeddings_id.npz")

import os
import shutil
import random

# Thư mục chứa ảnh
face_dir = "../data/images_face"
train_dir = "../data/faces_train"
test_dir = "../data/faces_test"

# Tạo thư mục train và test nếu chúng không tồn tại
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Lặp qua các thư mục con trong thư mục faces
for folder in os.listdir(face_dir):
    face_path = os.path.join(face_dir, folder)

    # Lấy danh sách các tệp trong thư mục
    files = os.listdir(face_path)

    # Tính số lượng tệp trong thư mục
    num_files = len(files)

    # Tạo danh sách các tệp cho thư mục train và test
    train_files = random.sample(files, int(num_files * 0.8))
    test_files = [f for f in files if f not in train_files]

    # Sao chép các tệp vào thư mục train và test
    for f in train_files:
        src = os.path.join(face_path, f)
        dst = os.path.join(train_dir, folder, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for f in test_files:
        src = os.path.join(face_path, f)
        dst = os.path.join(test_dir, folder, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

# Đếm số lượng tệp trong thư mục train và test
num_train_files = sum([len(files) for r, d, files in os.walk(train_dir)])
num_test_files = sum([len(files) for r, d, files in os.walk(test_dir)])

# In kết quả
print("Số lượng ảnh trong thư mục train là:", num_train_files)  # 2104
print("Số lượng ảnh trong thư mục test là:", num_test_files)  # 553

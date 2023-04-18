import os
import cv2
from deepface import DeepFace
import re


def extract_faces(img_path, save_path, file_name, detector_backend='mtcnn'):
    try:
        results = DeepFace.extract_faces(
            img_path=img_path,
            target_size=(224, 224),
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )
        if len(results) > 0:
            for result in results:
                if result['confidence'] > 0.9:
                    face = result['face']
                    face = (face * 255).astype('uint8')
                    rgb_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    os.makedirs(save_path, exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, file_name.split(".")[0] + ".jpg"), rgb_image)
    except Exception as e:
        print(f"Error: {e}")


def number_label(folder_name):
    match = re.search(r'\d+', folder_name)  # Tìm kiếm một chuỗi các số liên tiếp
    if match:
        number = int(match.group())  # Chuyển chuỗi số thành số nguyên
        return number
    return 0

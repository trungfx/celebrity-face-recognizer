from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def knn(x_train, y_train, x_test, y_test, k_num=None):
    if k_num:
        k_neighbors = [k_num]
    else:
        k_neighbors = [1, 3, 5, 7, 9, 11]
    performance_metrics = {}  # Tạo empty dictionary

    for k in k_neighbors:
        classifier = KNeighborsClassifier(metric="cosine", n_neighbors=k)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        # Tính toán các độ đo hiệu suất
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Thêm các giá trị vào dictionary với khoá là k
        performance_metrics[k] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    return performance_metrics

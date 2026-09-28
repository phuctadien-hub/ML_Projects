# Bài 3.30: Phân lớp nhị phân bằng Perceptron
# Dữ liệu: Breast Cancer Wisconsin (có sẵn trong scikit-learn)
#   nhãn: 1 = lành tính, 0 = ác tính  (đổi sang +1 / -1 cho Perceptron)
# Cài thư viện nếu thiếu:  pip install numpy scikit-learn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Perceptron:
    def __init__(self, learning_rate=1.0, max_epochs=100, random_state=0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state

    def _them_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        X = self._them_bias(np.array(X, dtype=float))
        y = np.array(y)
        rng = np.random.RandomState(self.random_state)
        self.w = np.zeros(X.shape[1])
        for _ in range(self.max_epochs):
            so_sai = 0
            for i in rng.permutation(len(y)):
                if y[i] * np.dot(self.w, X[i]) <= 0:
                    self.w += self.learning_rate * y[i] * X[i]
                    so_sai += 1
            if so_sai == 0:
                break
        return self

    def predict(self, X):
        X = self._them_bias(np.array(X, dtype=float))
        return np.where(np.dot(X, self.w) >= 0, 1, -1)


def danh_gia(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred))


# 1. Đọc dữ liệu, đổi nhãn 0/1 -> -1/+1
data = load_breast_cancer()
X, y = data.data, np.where(data.target == 1, 1, -1)
print("Số mẫu:", X.shape[0], "| Số đặc trưng:", X.shape[1])

# 2. Chia train / validation / test = 60% / 20% / 20%
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

# 3. Chuẩn hoá dữ liệu (rất quan trọng với Perceptron)
scaler = StandardScaler().fit(X_train)
X_train, X_val, X_test = (scaler.transform(a) for a in (X_train, X_val, X_test))

# 4. Thử nhiều siêu tham số, chọn bộ có F1 tốt nhất trên tập validation
print("\nTìm siêu tham số tốt nhất (theo F1 trên validation):")
best_f1, best_params = -1, None
for lr in [0.001, 0.01, 0.1, 1.0]:
    for epochs in [10, 50, 100, 500]:
        m = Perceptron(lr, epochs).fit(X_train, y_train)
        f1 = f1_score(y_val, m.predict(X_val))
        print(f"  lr={lr:<6} epochs={epochs:<4} F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_params = f1, (lr, epochs)
print("=> Tốt nhất: lr = %s, epochs = %s (F1 val = %.4f)" % (*best_params, best_f1))

# 5. Huấn luyện lại bằng train + val, đánh giá trên tập test
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])
model = Perceptron(*best_params).fit(X_full, y_full)
y_pred = model.predict(X_test)

acc, pre, rec, f1 = danh_gia(y_test, y_pred)
print("\nKẾT QUẢ TRÊN TẬP TEST")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {pre:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

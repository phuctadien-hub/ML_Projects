# Bài 3.29: Xây dựng lớp Perceptron (có fit và predict)
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=1.0, max_epochs=100, random_state=0):
        self.learning_rate = learning_rate   # tốc độ học
        self.max_epochs = max_epochs         # số vòng lặp tối đa qua toàn bộ dữ liệu
        self.random_state = random_state
        self.w = None                        # trọng số (phần tử cuối là bias)
        self.errors_ = []                    # số mẫu sai ở mỗi epoch

    def _them_bias(self, X):
        """Thêm cột 1 vào cuối X để gộp bias vào w."""
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        """Huấn luyện. y phải có giá trị -1 hoặc +1."""
        X = self._them_bias(np.array(X, dtype=float))
        y = np.array(y)
        rng = np.random.RandomState(self.random_state)
        self.w = np.zeros(X.shape[1])
        self.errors_ = []

        for epoch in range(self.max_epochs):
            so_sai = 0
            for i in rng.permutation(len(y)):        # duyệt mẫu theo thứ tự ngẫu nhiên
                if y[i] * np.dot(self.w, X[i]) <= 0:  # phân lớp sai
                    self.w += self.learning_rate * y[i] * X[i]
                    so_sai += 1
            self.errors_.append(so_sai)
            if so_sai == 0:                           # không còn sai -> dừng sớm
                break
        return self

    def predict(self, X):
        """Dự báo nhãn -1 hoặc +1 cho dữ liệu mới."""
        X = self._them_bias(np.array(X, dtype=float))
        return np.where(np.dot(X, self.w) >= 0, 1, -1)


# ---- Chạy thử: cổng AND (tách tuyến tính được) ----
if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [-1, -1, -1, 1]

    model = Perceptron(learning_rate=1.0, max_epochs=20)
    model.fit(X, y)

    print("Trọng số w (gồm bias):", model.w)
    print("Số mẫu sai mỗi epoch :", model.errors_)
    print("Dự báo               :", model.predict(X))
    print("Nhãn thật            :", y)
    print("Dự báo mẫu mới [1, 1]:", model.predict([[1, 1]]))

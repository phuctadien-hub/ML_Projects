# Bài 3.27: Perceptron - tính w^T x và dự đoán nhãn
import numpy as np

w = np.array([1, 2, -10])   # đã gồm bias ở phần tử cuối
x = np.array([3, 4, 1])     # đã thêm bias (số 1 ở cuối)

# Câu 1: tính w^T x
z = np.dot(w, x)            # 1*3 + 2*4 + (-10)*1 = 1
print("Câu 1: w^T x =", z)

# Câu 2: nhãn dự đoán = sign(w^T x)
y_du_doan = 1 if z >= 0 else -1
print("Câu 2: nhãn dự đoán =", y_du_doan)

# Câu 3: nếu nhãn thực tế y = -1
y_thuc = -1
if y_du_doan != y_thuc:
    print("Câu 3: y thực =", y_thuc, "khác nhãn dự đoán -> điểm BỊ phân lớp sai")
else:
    print("Câu 3: điểm được phân lớp đúng")

# Bài 3.28: Một bước cập nhật Perceptron
import numpy as np

w = np.array([-2, 1, 0])
x = np.array([2, 3, 1])     # đã thêm bias
y = 1

# Câu 1: kiểm tra phân lớp sai. Sai khi y * (w^T x) <= 0
z = np.dot(w, x)            # -4 + 3 + 0 = -1
print("Câu 1: w^T x =", z, " ->  y * w^T x =", y * z)
sai = (y * z) <= 0
print("        Mẫu bị phân lớp sai?", sai)

# Câu 2: nếu sai thì cập nhật  w_moi = w + y * x
if sai:
    w_moi = w + y * x
    print("Câu 2: w mới =", w_moi)     # [0, 4, 1]
else:
    w_moi = w
    print("Câu 2: không cần cập nhật")

# Câu 3: tính lại w^T x
z_moi = np.dot(w_moi, x)              # 0 + 12 + 1 = 13
print("Câu 3: w^T x sau cập nhật =", z_moi)
print("        y * w^T x =", y * z_moi, "> 0 -> mẫu đã được phân lớp đúng")

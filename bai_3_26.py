# Bài 3.26: Gradient Descent cho f(x) = x^2 - 4x + 5

def f(x):
    """Hàm số cần tối thiểu hoá."""
    return x**2 - 4*x + 5

def df(x):
    """Đạo hàm: f'(x) = 2x - 4."""
    return 2*x - 4

x = 5.0        # điểm khởi tạo x(0)
eta = 0.2      # learning rate
so_buoc = 4

print("Câu 1: f'(x) = 2x - 4")
print()
print("Câu 2 + 3: 4 bước cập nhật  x(t+1) = x(t) - eta * f'(x(t))")
print(f"{'Bước':<6}{'x':<12}{'f(x)':<12}{'f prime(x)':<12}")
print(f"{0:<6}{x:<12.4f}{f(x):<12.4f}{df(x):<12.4f}")

for t in range(1, so_buoc + 1):
    x = x - eta * df(x)          # cập nhật x
    print(f"{t:<6}{x:<12.4f}{f(x):<12.4f}{df(x):<12.4f}")

print()
print("Câu 4: Nhận xét")
print("- x giảm dần: 5 -> 3.8 -> 3.08 -> 2.648 -> 2.3888, tiến về nghiệm x* = 2.")
print("- f(x) giảm dần: 10 -> 4.24 -> 2.1664 -> 1.4199 -> 1.1512, tiến về giá trị nhỏ nhất f(2) = 1.")
print("- Khoảng cách tới x* nhân với (1 - 2*eta) = 0.6 sau mỗi bước nên hội tụ.")
print("- |f'(x)| nhỏ dần về 0 (gần điểm cực tiểu, độ dốc gần bằng 0).")

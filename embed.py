import numpy as np
from numpy import linalg as la
from scipy.stats import entropy
import matplotlib.pyplot as plt
import cv2
import pywt
import os

if __name__ == '__main__':
    M = 512  #Sửa chiều dài và chiều rộng của ảnh gốc
    N = 128 # Đã sửa lỗi chiều dài và chiều rộng hình mờ
    K = 64 # Cắt kích thước khối

    alpha = 0.1  # Hệ số cường độ

    path = 'out'
    if not os.path.isdir(path):
        os.makedirs(path)

    origin_pictures = []
    
    watered_pictures = []
    for root, dirs, files in os.walk("origin_image"):
        for file in files:
            origin_pictures.append(file)


    for i in range(len(origin_pictures)):
        I = cv2.imread("./origin_image/" + origin_pictures[i])  # (origin picture)
        G = cv2.imread("ptit.png")  # (watermark)

        I = cv2.resize(I, (M, M))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)  

        G = cv2.resize(G, (N, N))
        G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
        G = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)  

        plt.subplot(2, 2, 1)
        plt.imshow(I, cmap="gray")
        plt.title("image")
        plt.subplot(2, 2, 2)
        plt.imshow(G, cmap="gray")
        plt.title("watermark")

        # Step 1
        LL, (LH, HL, HH) = pywt.dwt2(G, 'haar') # Thực hiện chuyển đổi 2D haar DWT
        [U, S, V] = la.svd(HH) # Thực hiện phân rã SVD trên HH và thu được ma trận U, S, V

        # Step 2
        # Thực hiện chuyển đổi haar DWT cấp 2
        LL1, (LH1, HL1, HH1) = pywt.dwt2(I, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')  # 128*128
        # Step 3
        # Chọn khối nhúng tốt nhất mặc định là 4 * 4: (1, 1)
        optimal_block_index = 0

        # Step 4
        # Thực hiện chuyển đổi DCT trên khối nhúng đã chọn để thu được ma trận hệ số DCT B
        m = np.floor(optimal_block_index / 4) + 1
        n = np.mod(optimal_block_index, 4) + 1
        x = (m - 1) * K + 1
        y = (n - 1) * K + 1
        H_I = HH2[int(x): int(x + K), int(y): int(y + K)]
        B = cv2.dct(np.float32(H_I))

        # Step 5
        # Thực hiện phân rã giá trị số ít trên B và nhúng hình mờ
        U1, S1, V1 = la.svd(B)
        S2 = S1 + alpha * S
        B1 = U1 * S2 * V1
        H_I = cv2.idct(B1)
        HH2[int(x): int(x + K), int(y):int(y + K)] = H_I
        LL1 = pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar')
        W = pywt.idwt2((LL1, (LH1, HL1, HH1)), 'haar')
        W = np.uint8(W)

        plt.subplot(2, 2, 3)
        plt.imshow(W, cmap="gray")
        plt.title('img_watermarked')

        plt.show()

        watered_pictures.append(W)

    # save all picture change to the file
    watered_path = './watered_image/'
    if not os.path.isdir(watered_path):
        os.makedirs(watered_path)

    for i in range(len(watered_pictures)):
        print(origin_pictures[i])
        cv2.imwrite(watered_path + "watered_" + origin_pictures[i], watered_pictures[i])
import numpy as np
from numpy import linalg as la
from scipy.stats import entropy
import matplotlib.pyplot as plt
import cv2
import pywt
import os


if __name__ == '__main__':
    M = 512  
    N = 128 
    K = 64 

    alpha = 0.1  

    origin_pictures = []
    watered_pictures = []
    recover_pictures = []
    for root, dirs, files in os.walk("origin_image"):
        for file in files:
            origin_pictures.append(file)
    for root, dirs, files in os.walk("watered_image"):
        for file in files:
            watered_pictures.append(file)

    for i in range(len(watered_pictures)):
        I = cv2.imread("./origin_image/" + origin_pictures[i])
        I = cv2.resize(I, (M, M))
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        
        G = cv2.imread("ptit.png") 
        G = cv2.resize(G, (N, N))
        G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
        G = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
        
        W = cv2.imread("./watered_image/" + watered_pictures[i])

        optimal_block_index = 0

        LL, (LH, HL, HH) = pywt.dwt2(G, 'haar')  
        [U, S, V] = la.svd(HH) 

        LL1, (LH1, HL1, HH1) = pywt.dwt2(I, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar') 

        m = np.floor(optimal_block_index / 4) + 1
        n = np.mod(optimal_block_index, 4) + 1
        x = (m - 1) * K + 1
        y = (n - 1) * K + 1
        H_I = HH2[int(x): int(x + K), int(y): int(y + K)]
        B = cv2.dct(np.float32(H_I))

        U1, S1, V1 = la.svd(B)

        LL3, (LH3, HL3, HH3) = pywt.dwt2(W, 'haar')
        LL4, (LH4, HL4, HH4) = pywt.dwt2(LL3, 'haar')  # 128*128
        H_I2 = HH4[int(x):int(x + K), int(y):int(y + K)]
        B2 = cv2.dct(np.float32(H_I2))
        Uw, Sw, Vw = la.svd(B2)
        Sx = (Sw - S1) / alpha
        B2 = U * Sx * V
        H_I2 = cv2.idct(B2)
        A = pywt.idwt2((LL, (LH, HL, H_I2)), 'haar')
        A = np.uint8(A)

        recover_pictures.append(A)

    extract_path = './extract_image/'
    if not os.path.isdir(extract_path):
        os.makedirs(extract_path)

    for i in range(len(recover_pictures)):
        cv2.imwrite(extract_path + "recovered_" + watered_pictures[i], recover_pictures[i])

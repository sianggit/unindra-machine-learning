import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Simulasi Operasi Konvolusi 2D Citra Input (6x6) dengan Filter Kernel Sobel (3x3)
np.random.seed(42)
citra_input = np.array([
    [10, 10, 10,  0,  0,  0],
    [10, 10, 10,  0,  0,  0],
    [10, 10, 10,  0,  0,  0],
    [10, 10, 10,  0,  0,  0],
    [10, 10, 10,  0,  0,  0],
    [10, 10, 10,  0,  0,  0]
], dtype=float)

# Filter Sobel Vertical Edge Detection
kernel_sobel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)

def konvolusi_2d(citra, kernel, stride=1, padding=0):
    if padding > 0:
        citra = np.pad(citra, padding, mode='constant')
    h_in, w_in = citra.shape
    k_h, k_w = kernel.shape
    h_out = int((h_in - k_h) / stride) + 1
    w_out = int((w_in - k_w) / stride) + 1
    out = np.zeros((h_out, w_out))
    for i in range(0, h_out):
        for j in range(0, w_out):
            r_start = i * stride
            c_start = j * stride
            sub_mat = citra[r_start:r_start+k_h, c_start:c_start+k_w]
            out[i, j] = np.sum(sub_mat * kernel)
    return out

def max_pooling_2d(f_map, pool_size=2, stride=2):
    h, w = f_map.shape
    h_out = int((h - pool_size) / stride) + 1
    w_out = int((w - pool_size) / stride) + 1
    pooled = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out):
            r = i * stride
            c = j * stride
            pooled[i, j] = np.max(f_map[r:r+pool_size, c:c+pool_size])
    return pooled

# Hitung Feature Map dan Pooling
feature_map = konvolusi_2d(citra_input, kernel_sobel, stride=1, padding=0)
f_map_relu = np.maximum(0, feature_map)
pooled_map = max_pooling_2d(f_map_relu, pool_size=2, stride=2)

print('\n' + '='*55)
print('HASIL EVALUASI OPERASI KONVOLUSI 2D & CNN (BAB 13)')
print('='*55)
print(f'Dimensi Citra Input Asli     : {citra_input.shape[0]} x {citra_input.shape[1]} piksel')
print(f'Dimensi Kernel Filter Sobel  : {kernel_sobel.shape[0]} x {kernel_sobel.shape[1]}')
print(f'Dimensi Feature Map Konvolusi: {feature_map.shape[0]} x {feature_map.shape[1]}')
print(f'Dimensi Setelah Max Pooling  : {pooled_map.shape[0]} x {pooled_map.shape[1]} (Subsampling 4x)')
print(f'Respon Tepi Vertikal Maksimum: {np.max(feature_map):.1f} unit aktivasi')
print('='*55)

os.makedirs('visual_output', exist_ok=True)
plt.style.use('dark_background')

# Gambar 1: Transformasi Konvolusi Citra -> Kernel -> Feature Map
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), facecolor='#0B0E14')
for ax in axes:
    ax.set_facecolor('#0B0E14')

axes[0].imshow(citra_input, cmap='gray')
axes[0].set_title('1. Citra Input (6x6)', color='#F1F5F9', fontsize=9.5, fontweight=600)
axes[0].axis('off')

axes[1].imshow(kernel_sobel, cmap='coolwarm')
axes[1].set_title('2. Filter Sobel (3x3)', color='#F1F5F9', fontsize=9.5, fontweight=600)
axes[1].axis('off')

axes[2].imshow(f_map_relu, cmap='inferno')
axes[2].set_title('3. ReLU Feature Map (4x4)', color='#F1F5F9', fontsize=9.5, fontweight=600)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('visual_output/13_cnn_convolution_filter_feature_map.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Gambar 2: Subsampling Max Pooling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8), facecolor='#0B0E14')
ax1.set_facecolor('#0B0E14')
ax2.set_facecolor('#0B0E14')

im1 = ax1.imshow(f_map_relu, cmap='magma')
ax1.set_title('Feature Map Sebelum Pooling (4x4)', color='#F1F5F9', fontsize=9.5, fontweight=600)
ax1.axis('off')
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(pooled_map, cmap='magma')
ax2.set_title('Setelah Max Pooling 2x2 (2x2)', color='#F1F5F9', fontsize=9.5, fontweight=600)
ax2.axis('off')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('visual_output/13_cnn_max_pooling_subsample.png', dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

print('[OK] Visual output Bab 13 berhasil disimpan di visual_output/')


import cv2 as cv
from nemo_mask import generate_nemo_mask
from scipy import io
from scipy.stats import norm
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 第一张图片不采用hsv色域分割，因为尾巴那儿颜色不太白，无法很好的区分(注意修改True or False)
fish_309 = False

# 数据加载与解析
if fish_309:
    mask = io.loadmat('mask.mat')['Mask']
else:
    nemo_ = cv.imread('date/315.bmp')       # 注意：根据选择的不同图片进行修改
    mask = generate_nemo_mask(nemo_)
sample = io.loadmat('sample.mat')['array_sample']
src_image = Image.open('date/315.bmp')      # 注意：根据选择的不同图片进行修改
RGB_img = np.array(src_image)
Gray_img = np.array(src_image.convert('L'))



# 根据Mask，获取ROI区域
Gray_ROI = (Gray_img * mask)/255
RGB_mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
RGB_ROI = (RGB_img * RGB_mask)/255

# 根据标签拆分数据
gray1 = []
gray2 = []
RGB1 = []
RGB2 = []

for i in range(len(sample)):
    if(sample[i][4]) == 1.:                                           # 数据第5位为标签
        gray1.append(sample[i][0])
        RGB1.append(sample[i][1:4])
    else:
        gray2.append(sample[i][0])
        RGB2.append(sample[i][1:4])

RGB1 = np.array(RGB1)
RGB2 = np.array(RGB2)

# 计算两类在数据中的占比，即先验概率
P_pre1 = len(gray1)/len(sample)
P_pre2 = 1-P_pre1

# 一维时，贝叶斯
# ------------------------------------------------------------------------------------#
# 数据为一维时(灰度图像)，用最大似然估计两个类别条件概率pdf的参数——标准差与均值
gray1_m = np.mean(gray1)
gray1_s = np.std(gray1)
gray2_m = np.mean(gray2)
gray2_s = np.std(gray2)
# print(gray1_s, gray2_s)

# 绘制最大似然估计出的类条件pdf
x = np.arange(0, 1, 1/1000)
gray1_pdf = norm.pdf(x, gray1_m, gray1_s)
gray2_pdf = norm.pdf(x, gray2_m, gray2_s)

# 创建窗口
plt.figure(0)
# 子图1，显示pdf
ax = plt.subplot(2, 1, 1)
ax.set_title('p(x|w)')
ax.plot(x, gray1_pdf, 'r', x, gray2_pdf, 'b')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
# 子图2，显示相乘的结果
ax1 = plt.subplot(2, 1, 2)
ax1.plot(x, P_pre1*gray1_pdf, 'r', x, P_pre2*gray2_pdf, 'b')
ax1.set_title('p(w)*p(x|w)')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')

# 用最大后验贝叶斯对灰度图像进行分割
gray_out = np.zeros_like(Gray_img)
for i in range(len(Gray_ROI)):
    for j in range(len(Gray_ROI[0])):
        if Gray_ROI[i][j] == 0:
            continue
        elif P_pre1*norm.pdf(Gray_ROI[i][j], gray1_m, gray1_s) > P_pre2*norm.pdf(Gray_ROI[i][j], gray2_m, gray2_s):   # 贝叶斯公式分子比较
            gray_out[i][j] = 100
        else:
            gray_out[i][j] = 255

# plt.imshow(RGB_ROI)
plt.figure(1)
bx = plt.subplot(1, 1, 1)
bx.set_title('gray ROI')
bx.imshow(Gray_ROI, cmap='gray')
plt.figure(2)
bx1 = plt.subplot(1, 1, 1)
bx1.set_title('gray segment result')
bx1.imshow(gray_out, cmap='gray')

# 三维时，贝叶斯
# ------------------------------------------------------------------------------------#
# 数据为三维时(彩色图像)，用最大似然估计两个类别条件概率pdf的参数——协方差与均值
# 计算2个标签中3个通道的均值
RGB1_m = np.mean(RGB1, axis=0)
RGB2_m = np.mean(RGB2, axis=0)

cov_sum1 = np.zeros((3, 3))
cov_sum2 = np.zeros((3, 3))

for i in range(len(RGB1)):
    # print((RGB1[i]-RGB1_m).reshape(3, 1))
    cov_sum1 = cov_sum1 + np.dot((RGB1[i]-RGB1_m).reshape(3, 1), (RGB1[i]-RGB1_m).reshape(1, 3))

for i in range(len(RGB2)):
    cov_sum2 = cov_sum2 + np.dot((RGB2[i]-RGB2_m).reshape(3, 1), (RGB2[i]-RGB2_m).reshape(1, 3))

RGB1_cov = cov_sum1/(len(RGB1)-1)                      # 无偏估计除以N-1
RGB2_cov = cov_sum2/(len(RGB2)-1)

xx = np.array([x, x, x])
# print(P_pre1*multivariate_normal.pdf(RGB1, RGB1_m, RGB1_cov))

# 用最大后验贝叶斯对彩色图像进行分割
RGB_out = np.zeros_like(RGB_ROI)
for i in range(len(RGB_ROI)):
    for j in range(len(RGB_ROI[0])):
        if np.sum(RGB_ROI[i][j]) == 0:
            continue
        elif P_pre1*multivariate_normal.pdf(RGB_ROI[i][j], RGB1_m, RGB1_cov) > P_pre2*multivariate_normal.pdf(RGB_ROI[i][j], RGB2_m, RGB2_cov): # 贝叶斯公式分子比较
            RGB_out[i][j] = [255, 0, 0]
        else:
            RGB_out[i][j] = [255, 255, 255]
# print(RGB_ROI.shape)

# 显示RGB ROI，与彩色分割结果
plt.figure(3)
cx = plt.subplot(1, 1, 1)
cx.set_title('RGB ROI')
cx.imshow(RGB_ROI)
plt.figure(4)
cx1 = plt.subplot(1, 1, 1)
cx1.set_title('RGB segment result')
cx1.imshow(RGB_out)

plt.show()

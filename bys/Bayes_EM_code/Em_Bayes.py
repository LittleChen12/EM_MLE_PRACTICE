import os
from scipy import io
from scipy.stats import norm
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

# 文件夹，用于存放图片
plot_dir = 'EM_out'
if os.path.exists(plot_dir) == 0:
    os.mkdir(plot_dir)
# 数据加载与解析
mask = io.loadmat('mask.mat')['Mask']                 # 数据为一个字典，根据key提取数据
sample = io.loadmat('sample.mat')['array_sample']
src_image = Image.open('date/fish.bmp')
RGB_img = np.array(src_image)
Gray_img = np.array(src_image.convert('L'))

# 通过mask，获取ROI区域
Gray_ROI = (Gray_img * mask)/256
RGB_mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
RGB_ROI = (RGB_img * RGB_mask)/255

# 假设两类数据初始占比相同，即先验概率相同
P_pre1 = 0.5
P_pre2 = 0.5

# 假设每个数据来自两类的初始概率相同，即软标签相同
soft_guess1 = 0.5
soft_guess2 = 0.5

# 选择1维或是多维
#gray_status = True
gray_status = False

# 一维时的EM
# ----------------------------------------------------------------------------------------------------#
if gray_status:

    # 观察图像，肉眼估计初始值
    gray1_m = 0.5
    gray1_s = 0.1
    gray2_m = 0.8
    gray2_s = 0.3

    # 绘制假定的PDF
    x = np.arange(0, 1, 1/1000)
    gray1_pdf = norm.pdf(x, gray1_m, gray1_s)
    gray2_pdf = norm.pdf(x, gray2_m, gray2_s)
    plt.figure(0)
    ax = plt.subplot(1, 1, 1)
    ax.plot(x, gray1_pdf, 'r', x, gray2_pdf, 'b')
    ax.set_title('supposed PDF')

    plt.figure(1)
    ax1 = plt.subplot(1, 1, 1)
    ax1.imshow(Gray_ROI, cmap='gray')
    ax1.set_title('gray ROI')
    plt.show()

    gray = np.zeros((len(sample), 5))
    gray_s_old = gray1_s + gray2_s
    count = 0
    # 迭代更新参数
    for epoch in range(60):
        count = count + 1
        print(count)
        for i in range(len(sample)):

            # 贝叶斯计算每个数据的后验，即得到软标签
            soft_guess1 = (P_pre1*norm.pdf(sample[i][0], gray1_m, gray1_s))/(P_pre1*norm.pdf(sample[i][0], gray1_m, gray1_s) +
                                                                             P_pre2*norm.pdf(sample[i][0], gray2_m, gray2_s))
            soft_guess2 = 1 - soft_guess1
            gray[i][0] = sample[i][0]
            gray[i][1] = soft_guess1*1                         # 当前一个数据中类别1占的个数，1*后验，显然是小数
            gray[i][2] = soft_guess2*1
            gray[i][3] = soft_guess1*sample[i][0]              # 对当前数据中属于类别1的部分，当前数据*后验
            gray[i][4] = soft_guess2*sample[i][0]

            # 根据软标签，再借助最大似然估计出类条件概率PDF参数——均值，标准差

        gray1_num = sum(gray)[1]                                # 对每一个数据中类别1占的个数求和，就得到数据中类别1的总数
        gray2_num = sum(gray)[2]
        gray1_m = sum(gray)[3]/gray1_num                        # 对每一个数据中属于类别1的那部分求和，就得到类别1的x的和，用其除以类别1的个数就得到其均值
        gray2_m = sum(gray)[4]/gray2_num

        sum_s1 = 0.0
        sum_s2 = 0.0

        for i in range(len(gray)):
            sum_s1 = sum_s1 + gray[i][1]*(gray[i][0] - gray1_m)*(gray[i][0] - gray1_m)     # 每个数据的波动中，属于类别1的部分
            sum_s2 = sum_s2 + gray[i][2]*(gray[i][0] - gray2_m)*(gray[i][0] - gray2_m)
        gray1_s = pow(sum_s1/gray1_num, 0.5)                                               # 标准差
        gray2_s = pow(sum_s2/gray2_num, 0.5)

        # print(gray1_m, gray2_m, gray1_s, gray2_s)
        P_pre1 = gray1_num/(gray1_num + gray2_num)                                         # 更新先验概率
        P_pre2 = 1 - P_pre1

        gray1_pdf = norm.pdf(x, gray1_m, gray1_s)
        gray2_pdf = norm.pdf(x, gray2_m, gray2_s)
        gray_s_d = abs(gray_s_old - gray2_s - gray1_s)
        gray_s_old = gray2_s + gray1_s
        # if gray_s_d < 0.0001:                                                               # 迭代停止条件，如果两次方差变化较小则停止迭代
        #     break

        # 绘制更新参数后的pdf
        plt.figure(2)
        ax2 = plt.subplot(1, 1, 1)
        ax2.plot(x, gray1_pdf, 'r', x, gray2_pdf, 'b')
        ax2.set_title('epoch' + str(epoch + 1) + ' PDF')
        plt.savefig(plot_dir + '//' + 'PDF_' + str(epoch + 1) + '.jpg', dpi=100)
        plt.close()
        # plt.show()

        if epoch % 1 == 0:                                # 迭代2次进行一次分割测试

            gray_out = np.zeros_like(Gray_img)
            for i in range(len(Gray_ROI)):
                for j in range(len(Gray_ROI[0])):
                    if Gray_ROI[i][j] == 0:
                        continue
                    # 贝叶斯公式分子比较，等价于最大后验
                    elif P_pre1 * norm.pdf(Gray_ROI[i][j], gray1_m, gray1_s) > P_pre2 * norm.pdf(Gray_ROI[i][j],
                                                                                                 gray2_m, gray2_s):
                        gray_out[i][j] = 100
                    else:
                        gray_out[i][j] = 255
            # 显示分割结果
            plt.ion()
            plt.figure(3)
            ax3 = plt.subplot(1, 1, 1)
            ax3.imshow(gray_out, cmap='gray')
            ax3.set_title('epoch' + str(epoch + 1) + 'gray segment')
            plt.savefig(plot_dir + '//' + 'Gray_segment_' + str(epoch + 1) + '.jpg', dpi=100)
            plt.pause(0.1)  # 暂停0.1秒以更新图形

# 三维时的EM
# -------------------------------------------------------------------------------------------------------#
else:
    # 观察图像，肉眼估计初始值,为了显示迭代，此处估计将均值与方差设置的比较不符合
    RGB1_m = np.array([0.3, 0.3, 0.3])
    RGB2_m = np.array([0.1, 0.1, 0.1])
    RGB1_cov = np.array([[0.1, 0.04, 0.03],
                        [0.04, 0.1, 0.02],
                        [0.03, 0.02, 0.1]])
    RGB2_cov = np.array([[0.3, 0.02, 0.02],
                        [0.02, 0.3, 0.02],
                        [0.02, 0.02, 0.3]])

    RGB = np.zeros((len(sample), 11))

    # 显示彩色ROI
    plt.figure(3)
    cx = plt.subplot(1, 1, 1)
    cx.set_title('RGB ROI')
    cx.imshow(RGB_ROI)
    plt.show()
    # 迭代更新参数
    for epoch in range(30):
        for i in range(len(sample)):

            # 贝叶斯计算每个数据的后验，即得到软标签
            soft_guess1 = P_pre1*multivariate_normal.pdf(sample[i][1:4], RGB1_m, RGB1_cov)/(P_pre1*multivariate_normal.pdf(sample[i][1:4], RGB1_m, RGB1_cov) + P_pre2*multivariate_normal.pdf(sample[i][1:4], RGB2_m, RGB2_cov))
            soft_guess2 = 1 - soft_guess1
            RGB[i][0:3] = sample[i][1:4]
            RGB[i][3] = soft_guess1*1
            RGB[i][4] = soft_guess2*1
            RGB[i][5:8] = soft_guess1*sample[i][1:4]
            RGB[i][8:11] = soft_guess2*sample[i][1:4]
        # print(RGB[0])

        # 根据软标签，再借助最大似然估计出类条件概率PDF参数——均值，标准差
        RGB1_num = sum(RGB)[3]
        RGB2_num = sum(RGB)[4]
        RGB1_m = sum(RGB)[5:8]/RGB1_num
        RGB2_m = sum(RGB)[8:11]/RGB2_num

        # print(RGB1_num+RGB2_num, RGB1_m, RGB2_m)
        cov_sum1 = np.zeros((3, 3))
        cov_sum2 = np.zeros((3, 3))

        for i in range(len(RGB)):
            # print(np.dot((RGB[i][0:3]-RGB1_m).reshape(3, 1), (RGB[i][0:3]-RGB1_m).reshape(1, 3)))
            cov_sum1 = cov_sum1 + RGB[i][3]*np.dot((RGB[i][0:3]-RGB1_m).reshape(3, 1), (RGB[i][0:3]-RGB1_m).reshape(1, 3))
            cov_sum2 = cov_sum2 + RGB[i][4]*np.dot((RGB[i][0:3]-RGB2_m).reshape(3, 1), (RGB[i][0:3]-RGB2_m).reshape(1, 3))
        RGB1_cov = cov_sum1/(RGB1_num-1)                                                    # 无偏估计除以N-1
        RGB2_cov = cov_sum2/(RGB2_num-1)

        P_pre1 = RGB1_num/(RGB1_num + RGB2_num)
        P_pre2 = 1 - P_pre1

        print(RGB1_cov, P_pre1)

        # 用贝叶斯对彩色图像进行分割

        RGB_out = np.zeros_like(RGB_ROI)

        for i in range(len(RGB_ROI)):
            for j in range(len(RGB_ROI[0])):
                if np.sum(RGB_ROI[i][j]) == 0:
                    continue
                # 贝叶斯公式分子比较
                elif P_pre1 * multivariate_normal.pdf(RGB_ROI[i][j], RGB1_m, RGB1_cov) > P_pre2 * multivariate_normal.pdf(
                        RGB_ROI[i][j], RGB2_m, RGB2_cov):
                    RGB_out[i][j] = [155, 20, 0]
                else:
                    RGB_out[i][j] = [0, 155, 155]
        # print(RGB_ROI.shape)

        # 显示彩色分割结果
        plt.figure(4)
        ax3 = plt.subplot(1, 1, 1)
        ax3.imshow(RGB_out)
        ax3.set_title('epoch' + str(epoch + 1) + ' RGB segment')
        plt.savefig(plot_dir + '//' + 'RGB_segment_' + str(epoch + 1) + '.jpg', dpi=100)
        plt.close()







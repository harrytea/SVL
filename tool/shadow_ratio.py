import os
import cv2
import numpy as np

def calculate_shadow_ratio(mask_path):
    """
    计算给定路径下的 mask 图片的阴影占比（shadow_ratio）
    假设 mask 图片是黑白的：0 为背景，1 为阴影
    """
    # 读取 mask 图片（假设是灰度图）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 将图片归一化到 [0, 1]
    mask = mask / 255.0
    
    # 计算阴影区域的比例（阴影 = 1）
    shadow_ratio = np.mean(mask)  # 计算图像中 1 的占比
    return shadow_ratio

def compute_statistics(mask_dir):
    """
    计算一个文件夹中所有 mask 图片的阴影占比统计
    输出：min, mean, median, 90%, 99% 阴影占比
    """
    shadow_ratios = []
    
    # 遍历文件夹中的所有图片
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            mask_path = os.path.join(mask_dir, filename)
            shadow_ratio = calculate_shadow_ratio(mask_path)
            shadow_ratios.append(shadow_ratio)
    
    # 将列表转换为 numpy 数组，方便统计
    shadow_ratios = np.array(shadow_ratios)
    
    # 统计最小值、平均值、中位数、90% 和 99% 分位数
    min_ratio = np.min(shadow_ratios)
    mean_ratio = np.mean(shadow_ratios)
    median_ratio = np.median(shadow_ratios)
    percentile_90 = np.percentile(shadow_ratios, 90)
    percentile_99 = np.percentile(shadow_ratios, 99)
    
    return min_ratio, mean_ratio, median_ratio, percentile_90, percentile_99

# 设置你的 mask 文件夹路径
mask_dir = '/SSD/wangyh/shadow/shadowdata/SBU-shadow/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks'

# 获取统计结果
min_ratio, mean_ratio, median_ratio, percentile_90, percentile_99 = compute_statistics(mask_dir)

# 输出统计结果
print(f"Shadow Ratio Statistics:")
print(f"Min: {min_ratio:.4f}")
print(f"Mean: {mean_ratio:.4f}")
print(f"Median: {median_ratio:.4f}")
print(f"90th Percentile: {percentile_90:.4f}")
print(f"99th Percentile: {percentile_99:.4f}")
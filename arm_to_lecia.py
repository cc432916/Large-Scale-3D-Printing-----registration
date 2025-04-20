import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd

#part 1:机械臂坐标系下，热床的标定

def unit_vector(v):
    return v / np.linalg.norm(v)

def build_platform_frame(p1, p2, p3):
    x_axis = unit_vector(p2 - p1)
    z_axis = unit_vector(np.cross(p2 - p1, p3 - p1))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = unit_vector(y_axis)

    # 构造变换矩阵 R 和 T
    R_B = np.column_stack((x_axis, y_axis, z_axis))
    T_B = np.eye(4)
    T_B[:3, :3] = R_B
    T_B[:3, 3] = p1  # 改为平台中心点
    return T_B

# 从TXT文件读取路径点，每行格式为: x y z
def read_path_from_txt(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append([float(p) for p in parts[:3]])
    return np.array(points)

#将路径点保存到txt文件中
def save_path_to_txt(points, save_path):
    with open(save_path, 'w') as f:
        for p in points:
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")


#part 2:机械臂在全局坐标系下的标定 and 机械臂坐标系与全局坐标系的相互转换

def compute_transformation_matrix(p1, p2, p3):
    #根据三个非共线点计算从机械臂坐标系到打印平台坐标系的齐次变换矩阵。T: 4x4齐次变换矩阵，可将机械臂坐标系中的点转换到打印平台坐标系
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # 计算平面向量和法向量
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    Z_axis = normal / np.linalg.norm(normal)
    # 计算X轴（沿p1->p2方向）
    X_axis = v1 / np.linalg.norm(v1)

    # 计算Y轴（Z × X，确保右手坐标系）
    Y_axis = np.cross(Z_axis, X_axis)
    Y_axis /= np.linalg.norm(Y_axis)
    # 构建旋转矩阵（新坐标系基向量作为行向量）
    rotation_matrix = np.array([X_axis, Y_axis, Z_axis])

    # 计算平移向量（新坐标系原点在机械臂坐标系中的坐标的负投影）
    translation = -rotation_matrix @ p1

    # 构建齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation

    return T


# 利用对应点对求取D → G变换矩阵
def compute_D_to_G_transform(P_D_list, P_G_list):
    P_D = np.array(P_D_list)
    P_G = np.array(P_G_list)
    centroid_D = np.mean(P_D, axis=0)
    centroid_G = np.mean(P_G, axis=0)
    H = (P_D - centroid_D).T @ (P_G - centroid_G)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_G - R @ centroid_D
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def transform_points(points, transform_matrix):
    """
    批量转换点集坐标系

    参数:
        trans_matrix: 4x4齐次变换矩阵
        points: Nx3点集合

    返回:
        transformed_points: 转换后的Nx3点集合
    """
    points = np.array(points)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))
    transformed = (transform_matrix @ points_h.T).T
    return transformed[:, :3]
# Rodrigues构造旋转矩阵
def rodrigues_rotation(a, b, c):
    S = np.array([[0, -c, b],
                  [c, 0, -a],
                  [-b, a, 0]])
    I = np.eye(3)
    R = np.linalg.inv(I - S) @ (I + S)
    return R

# 最小二乘误差函数
def error_function(params, source, target):
    a, b, c, dx, dy, dz = params
    R = rodrigues_rotation(a, b, c)
    T = np.array([dx, dy, dz])
    transformed = (R @ source.T).T + T
    return np.sum(np.linalg.norm(transformed - target, axis=1) ** 2)

# 最小二乘法拟合变换（Rodrigues法）
def least_squares_fit(src, dst):
    initial_guess = np.zeros(6)
    result = minimize(error_function, initial_guess, args=(src, dst), method='L-BFGS-B')
    a, b, c, dx, dy, dz = result.x
    R = rodrigues_rotation(a, b, c)
    T = np.array([dx, dy, dz])
    return R, T

def svd_fit(src, dst):
    """SVD闭式解法求解刚体变换"""
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = dst_mean - R @ src_mean
    return R, T

def apply_transform(points, R, T):
    """应用旋转和平移到点云"""
    return (R @ points.T).T + T

def evaluate_error(transformed, target):
    """误差评估（均方根误差）"""
    errors = np.linalg.norm(transformed - target, axis=1)
    return errors, np.mean(errors), np.max(errors)


def batch_transform(T, points):
    points = np.asarray(points)
    if points.shape[1] == 3:
        # 点是三维坐标 (x, y, z)，需要加一列1
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    elif points.shape[1] == 4:
        # 已经是齐次坐标，不需要再加
        points_h = points
    else:
        raise ValueError("输入点的维度必须是3或4，当前是 %d 维" % points.shape[1])
    return (T @ points_h.T).T[:, :3]

#  可视化3D路径
def visualize_3d_path(points, title='3D Path in G coordinate'):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.asarray(points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o', color='blue')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
# 验证测试
if __name__ == "__main__":
    # 输入输出路径
    input_txt_path = r"D:\23Mechanical Engineering\laboratory\my paper\measure\path_d.txt"
    output_txt_path = r"D:\23Mechanical Engineering\laboratory\my paper\measure\path_g.txt"
    # 输入输出路径
    P_D_example = [
        [100, 200, 300],
        [150, 250, 300],
        [200, 300, 350],
        [250, 350, 400],
    ]
    P_G_example = [
        [1100, 2200, 1300],
        [1150, 2250, 1300],
        [1200, 2300, 1350],
        [1250, 2350, 1400],
    ]

    # 3. 计算D→G变换矩阵
    T_D_to_G = compute_D_to_G_transform(P_D_example, P_G_example)

    # 4. 读取机械臂轨迹（D系）
    path_D = read_path_from_txt(input_txt_path)

    # 5. 坐标变换（D → G）
    path_G = batch_transform(T_D_to_G, path_D)

    # 6. 保存转换后的路径
    save_path_to_txt(path_G, output_txt_path)

    # 7. 可视化路径（G系）
    visualize_3d_path(path_G, title="Transformed Path in G System")

    print("T_D_to_G 齐次变换矩阵:")
    print(np.round(T_D_to_G, 4))

    # -----------测试刚体变换鲁棒性-----------
    # 生成测试数据（已知变换矩阵）
    true_rotation = np.array([[0.866, -0.5, 0],
                              [0.5, 0.866, 0],
                              [0, 0, 1]])
    true_translation = np.array([1, 2, 3])
    # 创建源点云（机械臂坐标系）
    src_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    # 生成目标点云（激光跟踪仪坐标系）
    dst_points = apply_transform(src_points, true_rotation, true_translation)
    dst_points += np.random.normal(0, 0.001, dst_points.shape)  # 添加小噪声

    # 使用最小二乘拟合
    R_est, T_est = least_squares_fit(src_points, dst_points)
    print("估计的旋转矩阵 R：\n", np.round(R_est, 4))
    print("估计的平移向量 T：", np.round(T_est, 4))

    # 应用变换并计算误差
    transformed = apply_transform(src_points, R_est, T_est)
    err, mean_err, max_err = evaluate_error(transformed, dst_points)
    print(f"\n转换误差(mm): {np.round(err * 1000, 3)}")
    print(f"均方误差: {mean_err * 1000:.3f} mm，最大误差: {max_err * 1000:.3f} mm")

    # 验证转换精度
    T_est_homo = np.eye(4)
    T_est_homo[:3, :3] = R_est
    T_est_homo[:3, 3] = T_est
    transformed = batch_transform(T_est_homo, src_points)
    errors = np.linalg.norm(transformed - dst_points, axis=1)
    print("\n转换误差(mm):", np.round(errors * 1000, 4))

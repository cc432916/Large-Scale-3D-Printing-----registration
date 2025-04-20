import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('TkAgg')  # 在import pyplot之前指定后端
import matplotlib.pyplot as plt

class QuaternionKalmanFilter:
    def __init__(self, dt=0.01):
        # 状态维度: [x, y, z, vx, vy, vz, q0, q1, q2, q3]
        self.dt = dt  # 采样时间
        self.state_dim = 13
        self.obs_dim = 6  # 观测维度(x,y,z,rx,ry,rz)

        # 初始化状态向量
        self.x = np.zeros(self.state_dim)
        self.x[6] = 1.0  # 四元数初始化为单位四元数

        # 状态协方差矩阵
        self.P = np.eye(self.state_dim) * 0.1

        # 过程噪声协方差
        self.Q = np.eye(self.state_dim)
        self.Q[:3, :3] *= 0.01  # 位置噪声
        self.Q[3:6, 3:6] *= 0.1  # 速度噪声
        self.Q[6:, 6:] *= 0.001  # 四元数噪声

    @staticmethod
    def euler_to_quat(rx, ry, rz):
        # 欧拉角转四元数 (ZYX顺序)
        cy = np.cos(rz * 0.5)
        sy = np.sin(rz * 0.5)
        cp = np.cos(ry * 0.5)
        sp = np.sin(ry * 0.5)
        cr = np.cos(rx * 0.5)
        sr = np.sin(rx * 0.5)

        q0 = cr * cp * cy + sr * sp * sy
        q1 = sr * cp * cy - cr * sp * sy
        q2 = cr * sp * cy + sr * cp * sy
        q3 = cr * cp * sy - sr * sp * cy
        return np.array([q0, q1, q2, q3])

    def predict(self):
        # 状态转移矩阵 (匀速模型)
        F = np.eye(self.state_dim)
        F[:3, 3:6] = np.eye(3) * self.dt

        # 四元数运动学更新 (近似)
        omega = self.x[10:13]  # 假设有角速度状态
        dq = self.quat_exp(omega * self.dt)
        self.x[6:10] = self.quat_mult(self.x[6:10], dq)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])  # 归一化

        # 协方差预测
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, stds):
        # 观测值转换: 欧拉角转四元数
        pos_meas = z[:3]
        euler_meas = z[3:]
        q_meas = self.euler_to_quat(*euler_meas)

        # 构建观测向量 [x, y, z, q0, q1, q2, q3]
        H = np.zeros((7, self.state_dim))
        H[:3, :3] = np.eye(3)
        H[3:7, 6:10] = np.eye(4)

        # 动态观测噪声
        R = np.diag([stds[0] ** 2, stds[1] ** 2, stds[2] ** 2,
                     stds[3] ** 2, stds[4] ** 2, stds[5] ** 2, 0.001])

        # 卡尔曼增益
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        y = np.concatenate([pos_meas, q_meas])
        self.x += K @ (y - H @ self.x)
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    @staticmethod
    def quat_mult(q1, q2):
        # 四元数乘法
        q0 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q1_ = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q2_ = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q3_ = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        return np.array([q0, q1_, q2_, q3_])

    @staticmethod
    def quat_exp(w):
        # 四元数指数映射
        theta = np.linalg.norm(w)
        if theta < 1e-6:
            return np.array([1.0, 0, 0, 0])
        w_norm = w / theta
        return np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * w_norm])


# 示例使用 ------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化滤波器
    kf = QuaternionKalmanFilter(dt=0.01)  # dt需与数据采样间隔一致

    # 从文件加载实际数据 -----------------------------------------------------
    data_path = "your_measurement_data.csv"  # 修改为实际文件路径
    raw_data = np.loadtxt(data_path, delimiter=',')  # 假设数据格式为CSV

    # 数据格式要求：每行应包含 [x, δ1, y, δy, z, δz, rx, δrx, ry, δry, rz, δrz]
    # 共12列，顺序必须严格匹配

    filtered_results = []

    # 处理每一帧数据
    for row in raw_data:
        # 提取测量值和标准差
        z = np.array([row[0], row[2], row[4], row[6], row[8], row[10]])  # x,y,z,rx,ry,rz
        stds = np.array([row[1], row[3], row[5], row[7], row[9], row[11]])  # 各参数标准差

        # 单位转换（如果需要）
        z[3:] = np.radians(z[3:])  # 欧拉角转换为弧度

        # 滤波器执行
        kf.predict()
        kf.update(z, stds)

        # 保存结果
        filtered_results.append({
            'position': kf.x[:3].copy(),
            'quaternion': kf.x[6:10].copy(),
            'covariance': kf.P.diagonal().copy()
        })

    # 结果后处理 ------------------------------------------------------------
    # 将结果保存为文件
    np.savetxt("filtered_results.csv",
               [np.concatenate([r['position'], r['quaternion']]) for r in filtered_results],
               delimiter=',',
               header="x,y,z,q0,q1,q2,q3")

    # 可视化（示例显示X轴位置）
    plt.plot([r['position'][0] for r in filtered_results], label='Filtered X')
    plt.xlabel('Frame')
    plt.ylabel('X Position (mm)')
    plt.legend()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Qt5Agg')  # 或 'TkAgg'

# ------------------ 参数设置 ------------------
layer_thickness = 0.3
layers_per_segment = 17
input_path = r"D:/23Mechanical Engineering/laboratory/my paper/printing_test/20X30X40.gcode.txt"
transform_path = r"D:/23Mechanical Engineering/laboratory/my paper/printing_test/transformations.txt"
output_path = r"D:/23Mechanical Engineering/laboratory/my paper/printing_test/transformed_path.txt"

# ------------------ 路径读取 ------------------
def read_path_from_txt(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(data)

# ------------------ 分段逻辑 ------------------
def split_path_by_z(path, layer_thickness, layers_per_segment):
    z_min = np.min(path[:, 2])
    z_max = np.max(path[:, 2])
    segment_height = layer_thickness * layers_per_segment
    segments = []
    z_current = z_min
    while z_current <= z_max:
        mask = (path[:, 2] >= z_current) & (path[:, 2] < z_current + segment_height)
        segments.append(path[mask])
        z_current += segment_height
    return segments

# ------------------ 变换读取 ------------------
def read_transformations(filename):
    transformations = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith('# R'):
            R_mat = np.array([
                list(map(float, lines[i+1].split())),
                list(map(float, lines[i+2].split())),
                list(map(float, lines[i+3].split())),
            ])
            i += 4
        elif lines[i].startswith('# T'):
            T_vec = np.array(list(map(float, lines[i+1].split())))
            i += 2
            transformations.append((R_mat, T_vec))
        else:
            i += 1
    return transformations

# ------------------ 可视化 ------------------
def plot_segments_with_frames(segments):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, len(segments)))

    for i, segment in enumerate(segments):
        if len(segment) == 0:
            continue
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=colors[i], label=f"Segment {i+1}")

        # 坐标轴箭头
        origin = segment[-1]
        length = 20
        ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='r', length=length)
        ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='g', length=length)
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='b', length=length)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# ------------------ 主流程 ------------------
original_path = read_path_from_txt(input_path)
segments = split_path_by_z(original_path, layer_thickness, layers_per_segment)
transformations = read_transformations(transform_path)

transformed_segments = []

# 使用四元数旋转（防止数值误差积累）
cumulative_rotation = R.from_matrix(np.eye(3))
cumulative_translation = np.zeros(3)

for i, segment in enumerate(segments):
    if i == 0:
        transformed_segments.append(segment)
    else:
        if i - 1 < len(transformations):
            R_i, T_i = transformations[i - 1]
            R_step = R.from_matrix(R_i)

            # 累积旋转和平移
            cumulative_rotation = cumulative_rotation * R_step
            cumulative_translation = cumulative_rotation.apply(T_i) + cumulative_translation
        else:
            print(f"[警告] 第{i}段缺少变换参数，使用最后一次累计值")

        # 应用累积变换
        transformed_segment = cumulative_rotation.apply(segment) + cumulative_translation
        transformed_segments.append(transformed_segment)

# 输出变换后路径
all_transformed_points = np.vstack(transformed_segments)
with open(output_path, 'w') as f:
    for row in all_transformed_points:
        f.write(f'{row[0]:.3f} {row[1]:.3f} {row[2]:.3f}\n')
print(f"转换完成，保存至 {output_path}")

# 可视化
plot_segments_with_frames(transformed_segments)

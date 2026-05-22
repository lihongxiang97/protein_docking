"""
刚体变换几何工具：四元数旋转、欧拉角、碰撞检测。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 [w, x, y, z] 转 3x3 旋转矩阵。"""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def euler_to_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """ZYZ 欧拉角转旋转矩阵 (弧度)。"""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz1 @ Ry @ Rz2


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """均匀随机旋转 (Shoemake 方法)。"""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ])
    # scipy 格式 w,x,y,z -> 我们使用 w 在前
    w, x, y, z = q[3], q[0], q[1], q[2]
    return quaternion_to_matrix(np.array([w, x, y, z]))


def grid_rotations(n: int = 12) -> List[np.ndarray]:
    """生成均匀分布的旋转矩阵集合。"""
    matrices = []
    golden = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = 2 * math.pi * i / golden
        phi = math.acos(1 - 2 * (i + 0.5) / n)
        alpha, beta, gamma = theta, phi, theta * 0.5
        matrices.append(euler_to_matrix(alpha, beta, gamma))
    # 添加坐标轴 90° 旋转
    for axis in range(3):
        for angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            if axis == 0:
                matrices.append(euler_to_matrix(angle, 0, 0))
            elif axis == 1:
                matrices.append(euler_to_matrix(0, angle, 0))
            else:
                matrices.append(euler_to_matrix(0, 0, angle))
    return matrices[:n]


def apply_transform(
    coords: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """对坐标应用绕 center 旋转后平移。"""
    shifted = coords - center
    rotated = (rotation @ shifted.T).T
    return rotated + center + translation


def count_clashes(
    coords_a: np.ndarray,
    radii_a: np.ndarray,
    coords_b: np.ndarray,
    radii_b: np.ndarray,
    clash_cutoff: float = 2.0,
) -> int:
    """
    计算原子对碰撞数。
    两原子距离 < (r_a + r_b) * clash_factor 视为碰撞。
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0
    tree = cKDTree(coords_b)
    clash_count = 0
    for i, ca in enumerate(coords_a):
        ra = radii_a[i]
        neighbors = tree.query_ball_point(ca, ra + radii_b.max() + clash_cutoff)
        for j in neighbors:
            dist = np.linalg.norm(ca - coords_b[j])
            min_dist = ra + radii_b[j]
            if dist < min_dist * 0.85:  # 85% VdW 和视为严重碰撞
                clash_count += 1
    return clash_count


def count_contacts(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float = 5.0,
) -> int:
    """计算接触原子对数量。"""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0
    tree = cKDTree(coords_b)
    pairs = tree.query_ball_point(coords_a, cutoff)
    return sum(len(p) for p in pairs)


def compute_interface_area(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float = 5.0,
    atom_area: float = 15.0,
) -> float:
    """
    估算界面面积：接触原子数 × 平均原子贡献面积。
    简化 rolling sphere 思想的离散近似。
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0.0
    tree = cKDTree(coords_b)
    contacting = 0
    for ca in coords_a:
        dists, _ = tree.query(ca, k=min(5, len(coords_b)))
        if np.isscalar(dists):
            dists = [dists]
        if any(d < cutoff for d in dists):
            contacting += 1
    return contacting * atom_area

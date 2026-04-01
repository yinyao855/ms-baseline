from typing import List, Dict

import networkx as nx
import numpy as np
from collections import defaultdict


def cal_interactive_call_percentage(G, partitions: Dict[str, List[str]]):
    """
    计算跨分区调用百分比（ICP）：分区间调用占比的中位数（值越低越好）
    G: networkx有向图（节点为类，边为调用关系）
    partitions: 分区字典 {分区ID: [类1, 类2, ...]}
    """
    # 建立类到分区的映射 ~ O(V)
    class_to_part = {}
    for p_id, classes in partitions.items():
        for cls in classes:
            class_to_part[cls] = p_id

    sub_nodes = set(class_to_part.keys())
    subG = G.subgraph(sub_nodes)

    # 统计每对分区间的调用次数 ~ O(E)
    cross_calls = defaultdict(int)  # (p1, p2) -> 调用次数
    total_cross = 0  # 总跨分区调用次数

    for u, v in subG.edges():
        p_u = class_to_part[u]
        p_v = class_to_part[v]
        if p_u != p_v:
            cross_calls[(p_u, p_v)] += 1
            total_cross += 1

    if total_cross == 0:
        return 0.0  # 无跨分区调用时ICP为0

    # 计算每个跨分区调用对的占比 ~ O(M^2)
    ratios = [cnt / total_cross for cnt in cross_calls.values()]
    # 返回中位数（文档中采用中位数避免极端值影响）
    return np.median(ratios)


def cal_structure_modularity(G, partitions):
    """
    计算结构模块化（SM）：内聚性平均值 - 耦合性平均值（值越高越好）
    G: networkx有向图
    partitions: 分区字典
    """
    # 建立类到分区的映射及分区类数量 ~ O(V)
    class_to_part = {}
    part_sizes = {}  # 分区ID -> 类数量
    for p_id, classes in partitions.items():
        part_sizes[p_id] = len(classes)
        for cls in classes:
            class_to_part[cls] = p_id

    sub_nodes = set(class_to_part.keys())
    subG = G.subgraph(sub_nodes)

    partitions_list = list(partitions.keys())
    M = len(partitions_list)  # 总分区数

    # 1. 计算每个分区的内聚性（scoh）~ O(M*E)
    scoh_list = []
    for p_id in partitions_list:
        classes_p = partitions[p_id]
        m_i = part_sizes[p_id]
        if m_i <= 1:
            scoh_i = 0.0  # 单个类的分区内聚性为0
        else:
            # 统计分区内部的调用边数
            intra_edges = 0
            for u, v in subG.edges():
                if u in classes_p and v in classes_p:
                    intra_edges += 1
            scoh_i = intra_edges / (m_i ** 2)  # 内聚性公式
        scoh_list.append(scoh_i)
    avg_scoh = np.mean(scoh_list)  # 平均内聚性

    # 2. 计算每对分区的耦合性（scop）~ O(E + M^2)
    # 预先统计分区对之间的调用边数
    inter_edges_matrix = np.zeros((M, M), dtype=int)
    for u, v in subG.edges():
        if class_to_part[u] != class_to_part[v]:
            p_u = partitions_list.index(class_to_part[u])
            p_v = partitions_list.index(class_to_part[v])
            inter_edges_matrix[p_u][p_v] += 1

    scop_list = []
    for i in range(M):
        p1 = partitions_list[i]
        for j in range(i + 1, M):
            p2 = partitions_list[j]
            m1, m2 = part_sizes[p1], part_sizes[p2]
            if (m1 + m2) == 0:
                scop_ij = 0.0
            else:
                scop_ij = (inter_edges_matrix[i][j] + inter_edges_matrix[j][i]) / (2 * (m1 + m2))  # 耦合性公式
            scop_list.append(scop_ij)
    avg_scop = np.sum(scop_list) * 2 / (M * (M - 1)) if scop_list else 0.0  # 平均耦合性

    # 3. 计算SM
    return avg_scoh - avg_scop


def cal_interface_number(G, partitions):
    """
    计算平均接口数量（IFN）：每个分区对外接口数的平均值（值越低越好）
    G: networkx有向图（边为方法调用，此处简化为类级接口）
    partitions: 分区字典
    """
    # 建立类到分区的映射 ~ O(V)
    class_to_part = {}
    for p_id, classes in partitions.items():
        for cls in classes:
            class_to_part[cls] = p_id

    sub_nodes = set(class_to_part.keys())
    subG = G.subgraph(sub_nodes)

    # 统计每个分区的对外接口数（被其他分区调用的类）~ O(E)
    part_interfaces = defaultdict(set)  # 分区ID -> 对外接口类集合
    for u, v in subG.edges():
        p_u = class_to_part[u]
        p_v = class_to_part[v]
        if p_u != p_v:
            # v是p_v的类，被外部分区p_u调用，因此v是p_v的接口
            part_interfaces[p_v].add(v)

    # 计算每个分区的接口数（简化为被外部调用的类数量）~ O(M)
    ifn_list = [len(interfaces) for interfaces in part_interfaces.values()]
    # 若存在无外部调用的分区，接口数为0
    for p_id in partitions:
        if p_id not in part_interfaces:
            ifn_list.append(0)

    return np.mean(ifn_list)


def cal_non_extreme_distribution(partitions, min_threshold=5, max_threshold=20):
    """
    计算非极端分布（NED）：非极端分区的类占比（值越高越好）
    partitions: 分区字典
    min_threshold: 极端小分区的类数量阈值（默认5）
    max_threshold: 极端大分区的类数量阈值（默认20）
    """
    total_classes = sum(len(classes) for classes in partitions.values())
    if total_classes == 0:
        return 0.0

    # 统计非极端分区的类总数 ~ O(M)
    non_extreme_classes = 0
    for classes in partitions.values():
        cls_count = len(classes)
        if cls_count < min_threshold or cls_count > max_threshold:
            non_extreme_classes += cls_count

    return non_extreme_classes / total_classes


if __name__ == "__main__":
    # 1. 构建示例类调用图（有向图）
    G = nx.DiGraph()
    # 添加节点（类）
    example_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    G.add_nodes_from(example_classes)
    # 添加边（调用关系）：A->B表示A调用B
    edges = [
        ("A", "B"), ("B", "C"),  # 分区1内部调用
        ("C", "D"), ("D", "E"),  # 分区2内部调用
        ("A", "D"), ("B", "E"),  # 跨分区调用（1->2）
        ("F", "G"), ("G", "H"),  # 分区3内部调用
        ("H", "I"), ("I", "J"),  # 分区4内部调用
        ("F", "I"), ("G", "J")  # 跨分区调用（3->4）
    ]
    G.add_edges_from(edges)

    # 2. 定义示例分区结果
    example_partitions = {
        "1": ["A", "B", "C"],  # 3个类（非极端）
        "2": ["D", "E"],  # 2个类（极端小）
        "3": ["F", "G", "H"],  # 3个类（非极端）
        "4": ["I", "J"]  # 2个类（极端小）
    }

    # 3. 计算指标
    icp = cal_interactive_call_percentage(G, example_partitions)
    sm = cal_structure_modularity(G, example_partitions)
    ifn = cal_interface_number(G, example_partitions)
    ned = cal_non_extreme_distribution(example_partitions)

    # 4. 输出结果
    print(f"跨分区调用百分比（ICP）: {icp:.4f}")
    print(f"结构模块化（SM）: {sm:.4f}")
    print(f"平均接口数量（IFN）: {ifn:.4f}")
    print(f"非极端分布（NED）: {ned:.4f}")

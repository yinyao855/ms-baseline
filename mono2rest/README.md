# MONO2REST Baseline 实现

基于论文 [MONO2REST: Identifying and Exposing Microservices: a Reusable RESTification Approach](https://arxiv.org/abs/2503.21522) (arXiv:2503.21522v1) 的完整实现。

## 论文方法概述

MONO2REST 是一个两阶段框架，在 **不迁移** 单体系统的前提下，将其暴露为微服务应用：

### 第一阶段：微服务识别（方法级聚类）

1. **调用图提取** — 从单体系统中提取方法间有向调用关系
2. **方法语义嵌入** — 使用 SBERT (`bert-base-nli-mean-tokens`) 生成方法语义向量
3. **NSGA-III 多目标聚类** — 同时优化三个目标：
   - **最小化耦合度 (Coupling)**: `Coupling(x) = E_external(x) / (E_internal(x) + E_external(x))`
   - **最大化内聚度 (Cohesion)**: `Cohesion(x) = min(E_internal(x) / V_cluster(x), 1.0)`
   - **最大化语义相似度 (SemSim)**: `SemSim(x) = (1/V_cluster) × Σ Sim(i,j)`

### 第二阶段：REST API 生成

1. **暴露方法筛选** — 被其他集群调用的方法需要通过 API 暴露
2. **HTTP 方法分配** — 使用 Zero-Shot Classification (`facebook/bart-large-mnli`) 自动分类
3. **URI 生成** — 基于树结构，结合 NLTK POS 标注去除动词 + SBERT 语义合并类名

## 实现设计

### 与论文算法的对应关系

| 论文描述 | 实现文件 | 说明 |
|----------|----------|------|
| Call-Graph Extraction (§III-A1) | `ir_adapter.py` | 从 ir-a.json 的 `invokedMethods` 构建方法级调用图，替代论文中的 java-callgraph |
| Method Embedding (§III-A2) | `semantic_embedder.py` | 使用论文指定的 `bert-base-nli-mean-tokens` SBERT 模型 |
| NSGA-III Clustering (§III-A3) | `nsga_clustering.py` | 严格按论文公式实现三个目标函数，使用论文 Fig.5 的集群注入交叉和随机转移变异 |
| Method Selection (§III-B) | `rest_api_generator.py` | 筛选被其他集群调用的方法（论文 Fig.7） |
| HTTP Assignment (§III-B) | `rest_api_generator.py` | 使用 `facebook/bart-large-mnli` Zero-Shot 分类（论文 Fig.8） |
| URI Creation (§III-B) | `rest_api_generator.py` | 三步法：树初始化 → POS 标注去动词 → SBERT 类名合并（论文 Fig.9） |

### IR-A 适配层

本实现使用 OpenRewrite 提取的 `ir-a.json` 作为输入（替代论文中的 JAR 静态分析）：

- **调用图**：从每个方法的 `invokedMethods` 字段提取方法级调用关系
- **接口→实现映射**：当 `invokedMethods` 引用接口方法时，自动重定向到实现类
- **方法唯一标识**：使用 `classFqn#methodName` 格式（重载方法追加序号）
- **文本构建**：方法名 CamelCase 分词 + 参数名 + 类名

### 遗传算子实现

- **编码方式**：长度为 N 的整数数组，每个元素为集群分配 [0, K-1]
- **交叉算子**（论文 Fig.5）：从 parent1 选一个随机集群注入 parent2，修复重复方法
- **变异算子**：随机选一个方法转移到另一个集群
- **修复机制**：确保所有 K 个集群非空

### 方法级 → 类级映射

论文采用方法级聚类，但项目评估框架使用类级。转换规则：

- 统计每个类的方法在各集群中的分布
- 将类分配到包含其最多方法的集群
- 若方法分布比例 < 80%，标记为 `sharedClass`（策略：SPLIT）

## 所需模型

| 用途 | 模型 | 大小 |
|------|------|------|
| 语义嵌入 | `bert-base-nli-mean-tokens` (SBERT) | ~420MB |
| HTTP 方法分类 | `facebook/bart-large-mnli` (Zero-Shot) | ~1.6GB |
| POS 标注 | NLTK `averaged_perceptron_tagger` | ~2MB |

## 目录结构

```
mono2rest/
├── __init__.py              # 包初始化
├── data_models.py           # 数据模型: Method, Cluster, RESTEndpoint
├── ir_adapter.py            # IR-A 解析 + 调用图构建
├── semantic_embedder.py     # SBERT 嵌入 + 相似度矩阵
├── nsga_clustering.py       # NSGA-III 多目标聚类
├── rest_api_generator.py    # REST API 生成 (筛选 + HTTP分类 + URI)
├── main.py                  # 主入口 + CLI
├── run_demo.py              # Demo 脚本
└── README.md                # 本文档
```

## 安装

```bash
cd ms-baseline
uv sync --extra local
```

## 使用方法

```bash
# PetClinic，7 个集群
python -m mono2rest.main -i data/petclinic/ir-a.json -k 7

# DayTrader，5 个集群
python -m mono2rest.main -i data/daytrader/ir-a.json -k 5

# 自定义参数
python -m mono2rest.main -i data/petclinic/ir-a.json -k 7 -g 200 -p 150
```

### 在代码中使用

```python
from mono2rest.main import MONO2REST

config = {
    "num_clusters": 7,
    "max_generations": 100,
    "population_size": 100,
}
mono = MONO2REST(config)
result = mono.run("data/petclinic/ir-a.json")
```

### 运行 Demo

```bash
cd ms-baseline
python -m mono2rest.run_demo
python -m mono2rest.run_demo -i data/jpetstore/ir-a.json -k 5
```

## 输出格式

### 1. `mono2rest_result.json` — 方法级聚类 + REST endpoints

```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "methods": [{"id": "...#findPetById", "name": "findPetById", ...}],
      "metrics": {"coupling": 0.15, "cohesion": 0.82, "semantic_similarity": 0.89}
    }
  ],
  "rest_endpoints": [
    {"uri": "/clinic/pet/{petId}", "http_method": "GET", "method": {...}, "cluster_id": 1}
  ],
  "summary": {"total_methods": 57, "total_clusters": 7, "total_rest_endpoints": 12}
}
```

### 2. `clusters.json` — 类级分解

```json
{
  "relativePath": "spring-petclinic-server",
  "clusters": [
    {"id": 0, "name": "pet-service", "reason": "NSGA-III cluster 1", "classes": ["..."]}
  ],
  "sharedClasses": [
    {"fqn": "...ClinicServiceImpl", "strategy": "SPLIT", "reason": "...", "detail": "..."}
  ]
}
```

## 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_clusters` | 7 | 目标微服务数量 |
| `max_generations` | 100 | NSGA-III 进化代数 |
| `population_size` | 100 | 种群大小 |
| `crossover_rate` | 0.8 | 交叉概率 |
| `mutation_rate` | 0.1 | 变异概率 |

## 论文参考

```bibtex
@article{lecrivain2025mono2rest,
  title={MONO2REST: Identifying and Exposing Microservices: a Reusable RESTification Approach},
  author={Lecrivain, Matthéo and Barry, Hanifa and Tamzalit, Dalila and Sahraoui, Houari},
  journal={arXiv preprint arXiv:2503.21522},
  year={2025}
}
```

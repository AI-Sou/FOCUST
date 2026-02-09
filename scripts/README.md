# scripts

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



本目录提供面向 Linux 的 Bash 自动化脚本，用于将 FOCUST 的完整流程以可复现的方式串联执行。脚本按 `00_*.sh` 顺序编号，便于从数据集构建、训练、推理到评估与报告逐步推进。

Windows 与 macOS 场景建议使用 `gui.py` 与 `laptop_ui.py` 作为主入口。脚本会检测系统类型，在非 Linux 环境下会直接退出以避免误用。

---

## 运行方式

推荐直接用 bash 调用：

```bash
bash scripts/00_env_check.sh
```

如需直接执行脚本，可授予可执行权限：

```bash
chmod +x scripts/*.sh
chmod +x scripts/one_click/*.sh
```

---

## 两条流水线

### 经典 HCP 主链路

1. 数据集构建，生成统一的时序数据集结构
2. 训练二分类模型，用于菌落与非菌落过滤
3. 训练多分类模型，用于五类识别
4. 使用 `engine=hcp` 推理
5. 数据集评估与报告生成

### HCP 编码加 YOLO 可选链路

1. 将时序标注数据转换为 YOLO 所需的数据格式
2. 训练 YOLO 多菌落检测模型
3. 使用 `engine=hcp_yolo` 推理
4. 使用多分类模型对结果进行可选细化
5. 数据集评估与报告生成

---

## 脚本清单

- `00_env_check.sh` 环境自检
- `01_build_dataset_hcp.sh` 构建 HCP 推理与分类训练所需数据集
- `02_build_dataset_binary.sh` 构建二分类训练数据集
- `03_train_binary.sh` 二分类模型训练
- `04_train_multiclass.sh` 多分类模型训练
- `05_build_dataset_hcp_yolo.sh` 构建 HCP 编码加 YOLO 数据集
- `06_train_hcp_yolo.sh` 训练 YOLO 模型
- `07_detect_hcp.sh` 使用 `engine=hcp` 推理
- `08_detect_hcp_yolo.sh` 使用 `engine=hcp_yolo` 推理
- `09_evaluate_dataset.sh` 数据集评估
- `10_report_docx.sh` 生成 docx 报告

更细的参数说明以各脚本开头的 usage 文档为准。

---

## one_click

`scripts/one_click/` 提供将多个步骤串联的一键式脚本，适合在服务器上快速复现与批处理。入口文档见 `scripts/one_click/README.md`。

---

## 统一约定

- 仅支持 Linux，脚本会检查系统类型并在不满足条件时退出
- 脚本会自动切换到仓库根目录作为工作目录
- 默认使用 `python3`，可通过环境变量 `PYTHON` 指定解释器路径
- 系统默认离线运行，脚本默认从 `model/` 目录读取本地权重


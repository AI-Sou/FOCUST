# core

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>



`core/` 提供 FOCUST 的公共基础能力，包括配置管理、设备选择、训练调度封装、推理封装、时序处理与字体渲染。该目录被 `gui.py`、`laptop_ui.py` 以及工具脚本共同复用。

---

## 主要内容

- `config_manager.py`，配置管理，读取 `config/focust_config.json`
- `device_manager.py`，设备选择与统一抽象，支持 CPU、CUDA 与 Apple MPS
- `training_manager.py` 与 `training_wrappers.py`，训练任务封装，用于 GUI 调度
- `binary_inference.py`，二分类推理脚本
- `multiclass_inference.py`，多分类推理脚本
- `hcp_processor.py` 与 `sequence_processor.py`，时序处理与 HCP 编码相关逻辑
- `cjk_font.py`，中文字体的统一渲染封装

---

## 独立推理

二分类推理：

```bash
python core/binary_inference.py --model model/bi_cat98.pth --input /path/to/sequence_dir --device auto
python core/binary_inference.py --model model/bi_cat98.pth --input /path/to/sequence_dir --info
```

多分类推理：

```bash
python core/multiclass_inference.py --model model/multi_cat93.pth --input /path/to/sequence_dir --device auto
python core/multiclass_inference.py --model model/multi_cat93.pth --input /path/to/sequence_dir --info
```

如使用旧配置文件，更新配置中的权重路径字段，使其指向本地 `.pth` 权重即可。

---

## 中文字体

为避免图像与图表出现缺字方框，项目内置中文字体并提供统一封装：

- 字体文件：`assets/fonts/NotoSansSC-Regular.ttf`
- 入口脚本：`core/cjk_font.py`

如需使用自定义字体，可通过环境变量指定：

```bash
export FOCUST_CJK_FONT=/path/to/your_font.ttf
```

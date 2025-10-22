# RAG-HPO: 临床HPO语料匹配项目 (工程化重构版)

## 项目概述

RAG-HPO 是一个基于 Python 的工具，旨在从临床笔记中提取人类表型本体 (Human Phenotype Ontology, HPO) 术语。它利用大型语言模型 (LLM) 和检索增强生成 (Retrieval Augmented Generation, RAG) 技术，提供标准化的表型描述，这对于基因组学和临床研究至关重要。RAG-HPO 本身不是一个 LLM，但它利用用户配置的 LLM 来处理和标注临床文本。

**注意：保护患者信息并确保符合机构指南和 HIPAA 是最终用户的责任。**

## 工程化重构概览

为了将本项目从科研导向的 Jupyter Notebooks 转换为更具工程化、易于部署和维护的应用程序，我们进行了以下关键重构：

1.  **模块化**: 将核心逻辑从 Notebooks 中提取，并组织成结构化的 Python 模块。
2.  **命令行界面 (CLI)**: 引入 `click` 库，为知识库构建和 RAG 管道运行提供统一的命令行入口。
3.  **配置管理**: 将 LLM 相关的配置（如 API 密钥、模型名称）通过 CLI 参数或环境变量进行管理，而非交互式输入。
4.  **依赖管理**: 持续使用 Poetry 进行项目依赖管理，确保环境一致性。

这些改动使得项目更易于自动化、集成到其他系统，并为未来的功能扩展奠定了坚实的基础。

## 模块说明

本项目现在组织在 `rag_hpo/` 包中，包含以下核心模块：

*   **`rag_hpo/`**: 项目的根包目录。
*   **`rag_hpo/__init__.py`**: Python 包的初始化文件。
*   **`rag_hpo/utils.py`**:
    *   **作用**: 包含项目通用的辅助函数和类，如日志记录器 (`Logger`)、文本清理函数 (`clean_text_for_embedding`, `clean_clinical_note`) 以及状态管理（检查点）的辅助函数。
    *   **主要功能**: 提供统一的日志输出，确保文本预处理的一致性，并支持流程中断后的恢复。
*   **`rag_hpo/vectorization.py`**:
    *   **作用**: 负责 HPO 知识库的构建和更新。这是 RAG 流程中“检索”部分的基础。
    *   **主要功能**:
        *   自动下载 HPO OBO 本体文件。
        *   解析本体，提取 HPO 术语、定义、同义词、交叉引用 (SNOMED CT, UMLS) 和谱系信息。
        *   使用预训练的句子嵌入模型（如 SapBERT）将 HPO 术语及其相关信息向量化。
        *   将生成的元数据和向量嵌入保存为 `hpo_meta.json` 和 `hpo_embedded.npz` 文件。
*   **`rag_hpo/pipeline.py`**:
    *   **作用**: 实现 RAG 核心管道的逻辑，包括 LLM 交互、FAISS 检索和 HPO 术语的最终映射。
    *   **主要功能**:
        *   `LLMClient` 类：封装与 LLM API 的交互，包括令牌使用跟踪和速率限制。
        *   加载系统提示 (`system_prompts.json`)，指导 LLM 进行信息提取和术语映射。
        *   加载预构建的 HPO 向量数据库 (FAISS 索引)。
        *   处理临床笔记，通过 LLM 提取表型短语。
        *   利用 FAISS 检索与提取短语最相似的 HPO 候选术语。
        *   再次调用 LLM，从候选列表中选择最佳匹配的 HPO 术语。
        *   处理和输出最终的 HPO 标注结果。
*   **`rag_hpo/cli.py`**:
    *   **作用**: 项目的命令行接口，通过 `click` 库提供用户友好的命令行操作。
    *   **主要功能**: 定义了 `build-kb` 和 `process` 两个子命令，作为整个工作流的入口。
*   **`rag_hpo/config.py`**:
    *   **作用**: 目前是占位符文件，未来可用于集中管理项目配置，例如 LLM 默认参数、文件路径等。

## 使用说明

### 环境设置

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/your-repo/RAG-HPO.git
    cd RAG-HPO
    ```
2.  **安装依赖**: 确保您已安装 Poetry。然后运行：
    ```bash
    poetry install
    ```
    这将安装所有项目依赖，并设置好虚拟环境。

### 构建知识库

在运行 RAG 管道之前，您需要构建或更新 HPO 知识库。这包括下载 HPO 本体文件，并将其向量化。

```bash
poetry run rag-hpo build-kb [OPTIONS]
```

**常用选项**:

*   `--obo-url TEXT`: HPO OBO 文件的下载 URL (默认: `https://purl.obolibrary.org/obo/hp.obo`)。
*   `--obo-path TEXT`: 本地保存 HPO OBO 文件的路径 (默认: `hp.obo`)。
*   `--refresh-days INTEGER`: 如果本地 OBO 文件超过指定天数，则重新下载 (默认: `14`)。
*   `--meta-output TEXT`: HPO 元数据 JSON 文件的输出路径 (默认: `hpo_meta.json`)。
*   `--vec-output TEXT`: HPO 嵌入向量 NPZ 文件的输出路径 (默认: `hpo_embedded.npz`)。
*   `--hpo-full-csv TEXT`: 完整 HPO 术语 CSV 文件的输出路径 (默认: `hpo_terms_full.csv`)。
*   `--chpo-path PATH`: 可选的 CHPO Excel/CSV 路径，用于注入中文翻译。
*   `--use-sbert / --no-sbert`: 是否使用 SBERT 模型进行嵌入 (默认: `True`)。
*   `--sbert-model TEXT`: SBERT 模型名称 (默认: `pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb`)。
*   `--bge-model TEXT`: BGE 模型名称 (默认: `BAAI/bge-small-en-v1.5`)。
*   `--limit INTEGER`: 限制处理的 HPO 术语数量 (仅用于测试)。

**示例**:

```bash
# 构建一个用于测试的有限知识库，并保存到 data/ 目录下
poetry run rag-hpo build-kb \
    --obo-path data/hp.obo \
    --meta-output data/hpo_meta.json \
    --vec-output data/hpo_embedded.npz \
    --hpo-full-csv data/hpo_terms_full.csv \
    --chpo-path CHPO-2025-4.xlsx \
    --limit 100
```

针对中文生产环境，推荐直接切换到中文优化的 BGE 嵌入模型，并为输出与日志使用独立命名，示例命令如下：

```bash
mkdir -p logs
poetry run rag-hpo build-kb \
    --obo-path data/hp.obo \
    --meta-output data/hpo_meta_bge_small_zh.json \
    --vec-output data/hpo_embedded_bge_small_zh.npz \
    --hpo-full-csv data/hpo_terms_full_bge_small_zh.csv \
    --chpo-path CHPO-2025-4.xlsx \
    --no-sbert \
    --bge-model BAAI/bge-small-zh-v1.5 \
    2>&1 | tee logs/build_kb_bge_small_zh_$(date +%Y%m%d).log
```

### 运行 RAG 管道

使用已构建的知识库处理临床笔记，并提取 HPO 术语。

```bash
poetry run rag-hpo process [OPTIONS]
```

**常用选项**:

*   `--input-csv PATH`: 包含临床笔记的输入 CSV 文件路径 (必需)。
*   `--output-csv PATH`: 保存 HPO 标注结果的 CSV 文件路径。
*   `--output-json-raw PATH`: 保存 LLM 原始 JSON 输出的 CSV 文件路径。
*   `--display / --no-display`: 是否在终端显示结果 (默认: `False`)。
*   `--meta-path TEXT`: HPO 元数据 JSON 文件的路径 (默认: `hpo_meta.json`)。
*   `--vec-path TEXT`: HPO 嵌入向量 NPZ 文件的路径 (默认: `hpo_embedded.npz`)。
*   `--use-sbert / --no-sbert`: 是否使用 SBERT 模型进行嵌入 (默认: `True`)。
*   `--sbert-model TEXT`: SBERT 模型名称。
*   `--bge-model TEXT`: BGE 模型名称。
*   `--api-key TEXT`: LLM 的 API 密钥。**可以通过 `LLM_API_KEY` 环境变量设置。**
*   `--base-url TEXT`: LLM API 的基础 URL (默认: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`)。**可以通过 `LLM_BASE_URL` 环境变量设置。**
*   `--llm-model-name TEXT`: LLM 模型名称 (默认: `qwen3-max`)。**可以通过 `LLM_MODEL_NAME` 环境变量设置。**
*   `--max-tokens-per-day INTEGER`: LLM 每日最大令牌数 (默认: `500000`)。
*   `--max-queries-per-minute INTEGER`: LLM 每分钟最大查询数 (默认: `30`)。
*   `--temperature FLOAT`: LLM 的温度参数 (默认: `0.7`)。
*   `--system-prompts-file TEXT`: 系统提示 JSON 文件路径 (默认: `system_prompts.json`)。

**示例**:

```bash
# 假设您有一个名为 input.csv 的临床笔记文件，并且已经构建了知识库
# （默认使用阿里云 DashScope 的通义千问 Qwen3-Max 模型）
poetry run rag-hpo process \
    --input-csv input.csv \
    --output-csv output.csv \
    --meta-path data/hpo_meta_bge_small_zh.json \
    --vec-path data/hpo_embedded_bge_small_zh.npz \
    --display \
    --api-key "$LLM_API_KEY" \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
    --llm-model-name qwen3-max \
    --max-queries-per-minute 10 \
    --no-sbert \
    --bge-model BAAI/bge-small-zh-v1.5
```

### 通义千问 (Qwen) 快速接入

1.  在阿里云控制台获取 DashScope API Key，并设置到环境变量或直接通过命令行参数传入。例如在当前 shell 中执行：
    ```bash
    export LLM_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
    export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    export LLM_MODEL_NAME="qwen3-max"
    ```
    或者在 `rag-hpo process` 命令中逐一传递 `--api-key`、`--base-url`、`--llm-model-name`。
2.  运行示例命令：
    ```bash
    poetry run rag-hpo process \
        --input-csv data/test_cases_qwen.csv \
        --output-csv data/test_results_qwen.csv \
        --output-json-raw data/test_results_qwen_raw.csv \
        --meta-path data/hpo_meta_bge_small_zh.json \
        --vec-path data/hpo_embedded_bge_small_zh.npz \
        --api-key "$LLM_API_KEY" \
        --base-url "$LLM_BASE_URL" \
        --llm-model-name "$LLM_MODEL_NAME" \
        --max-queries-per-minute 10
    ```

> **提示**: 若使用 `source .env` 激活仓库中已配置好的 Poetry 环境，可直接在激活后的终端中运行上述命令。
**注意**: 对于敏感信息如 `API_KEY`，强烈建议通过环境变量设置，例如 `export LLM_API_KEY="YOUR_KEY"`。

仓库额外提供了一个中文病例示例 (`examples/clinical/case_phase2_input.csv`)，可直接用于验证中文知识库与 Qwen 接口的联调效果。

### Web 应用体验（MVP）

项目新增 `webapp/` 目录，提供粘贴式的交互界面，适合临床或医学部快速体验：

1. **启动 FastAPI 后端**（依赖与 CLI 相同）：
    ```bash
    poetry run uvicorn webapp.backend.app:app --reload --port 8000
    ```
    在启动前请确认环境变量 `LLM_API_KEY`、`HPO_META_PATH`、`HPO_VEC_PATH` 等已正确配置。
2. **启动 Vue 前端**：
    ```bash
    cd webapp/frontend
    npm install
    npm run dev
    ```
    默认使用 `http://localhost:8000` 作为 API 地址，如需更改可设置 `VITE_API_BASE_URL`。

前端支持在页面中粘贴单条临床描述、发起分析、对 HPO 结果进行复核，并一键导出 CSV；更详细的说明见 `webapp/README.md`。

## 未来改进

*   **更完善的配置管理**: 引入 `rag_hpo/config.py` 或使用 `PyYAML` 等库，支持从配置文件加载所有参数，提供更灵活的配置方式。
*   **错误处理与日志**: 进一步细化错误类型，提供更友好的错误信息和更详细的日志记录。
*   **测试套件**: 编写单元测试和集成测试，确保每个模块和整个管道的健壮性。
*   **性能优化**: 针对大规模数据处理，探索并行化或分布式处理方案。
*   **GUI 集成**: 考虑与现有或未来开发的 GUI 进行集成，提供更直观的用户体验。

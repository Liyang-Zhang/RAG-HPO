# RAG-HPO Web 应用 (MVP)

本目录包含一个供临床快速体验的 Web 端原型，包括：

- **backend/**：基于 FastAPI 的 HTTP 服务，直接复用 `rag_hpo` 的管线；
- **frontend/**：基于 Vue 3 + Vite 的单页应用，支持粘贴临床文本、启动分析、复核并导出结果。

## 运行前准备

1.  构建或下载 HPO 向量化资源（`data/hpo_meta_bge_small_zh.json`、`data/hpo_embedded_bge_small_zh.npz` 等）。
2.  准备 LLM 访问凭证，至少需要：
    ```bash
    export LLM_API_KEY="your_key"
    export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    export LLM_MODEL_NAME="qwen3-max"
    ```
    其他可选参数（如 `HPO_META_PATH`、`HPO_VEC_PATH` 等）详见 `webapp/backend/deps.py`。

## 启动后端

```bash
poetry install
poetry run uvicorn webapp.backend.app:app --reload --port 8000
```

接口:

- `GET /health`：返回当前配置；
- `POST /analyze`：JSON `{"text": "...临床描述..."}`，返回模型建议的 HPO 列表。

## 启动前端

```bash
cd webapp/frontend
npm install
npm run dev
```

默认端口为 `5173`，开发模式下会通过 Vite 代理将 API 请求转发到 `http://localhost:8000`。若后端端口不同，可在前端启动时设置 `VITE_API_BASE_URL`。

## 工作流程

1. 在页面中粘贴经脱敏的单条临床描述；
2. 点击「开始分析」等待结果返回；
3. 在表格中复核/编辑 HPO 映射，必要时调整类别或取消勾选；
4. 点击「导出 CSV」保存当前结果，或复制 JSON 原始输出用于进一步处理。

## 全流程示例

以下操作假设你已克隆仓库并位于项目根目录：

1. **准备向量库（首次使用）**
    ```bash
    poetry run rag-hpo build-kb \
      --obo-path data/hp.obo \
      --meta-output data/hpo_meta_bge_small_zh.json \
      --vec-output data/hpo_embedded_bge_small_zh.npz \
      --chpo-path CHPO-2025-4.xlsx \
      --no-sbert \
      --bge-model BAAI/bge-small-zh-v1.5
    ```
2. **导出必要环境变量（可写入 `.env` 或 shell 配置）**
    ```bash
    export LLM_API_KEY="sk-***"
    export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    export LLM_MODEL_NAME="qwen3-max"
    export HPO_META_PATH="data/hpo_meta_bge_small_zh.json"
    export HPO_VEC_PATH="data/hpo_embedded_bge_small_zh.npz"
    ```
3. **启动后端**
    ```bash
    poetry run uvicorn webapp.backend.app:app --reload --port 8000
    ```
    - 浏览器访问 `http://localhost:8000/health` 确认状态。
    - 可用 curl 触发一次分析：
      ```bash
      curl -X POST http://localhost:8000/analyze \
        -H "Content-Type: application/json" \
        -d '{"text": "诊断：原发闭经..."}'
      ```
4. **启动前端**
    ```bash
    cd webapp/frontend
    npm install
    npm run dev
    ```
    浏览器访问 `http://localhost:5173/`，粘贴临床文本即会调用 API。

## 常见问题

- **入口页面 404**：确认 `webapp/frontend/index.html` 存在，或使用 `npm run dev -- --host` 暴露局域网访问。
- **@ 别名失效**：`vite.config.ts` 已配置 `alias`，若 IDE 未识别，可确保 `tsconfig.json` 中的 `paths` 与之对应。
- **模型下载失败**：HuggingFace 可能网络不稳定，FastEmbed 会自动切换镜像；若持续失败，可预先下载模型或配置环境变量 `FASTEMBED_CACHE_BASE_DIR` 指向离线缓存。
- **LLM 认证错误**：后端会返回 400/401，前端提示“分析失败”；请重新确认 API Key 与 Base URL。

该 MVP 仅用于原型验证，不包含账号体系及批量处理。后续可以在此基础上扩充任务队列、审阅流程、角色权限等功能。

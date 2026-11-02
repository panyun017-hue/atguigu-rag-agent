# Atguigu Assistant 客服知识库 RAG 案例

一个基于 "Milvus + LangChain" 的简单 RAG（检索增强生成）机器人案例：把客服知识库文档切分、向量化后写入 Milvus，再根据用户问题检索相关片段，交由大模型生成回答。

本项目以 Jupyter Notebook 形式提供完整可运行的流程，适合学习 RAG 的基本链路：**文档加载 → 切分 → 向量化 → 入库 → 检索 → 生成**。

## 功能概览

- 使用 `TextLoader` 加载本地知识库 `knowledge.txt`
- 使用 `RecursiveCharacterTextSplitter` 按自定义分隔符切分文档
- 使用 BGE-M3（通过 SiliconFlow）生成 1024 维向量
- 使用 Milvus 存储向量并做 COSINE 相似度检索
- 使用 LangChain `create_agent` + 对话模型，基于检索上下文回答问题

## 项目结构
├── Atguigu Assistant客服知识库.ipynb # 主 Notebook，包含完整流程
├── knowledge.txt # 客服知识库文件（运行必需）
├── .env.example # 环境变量模板（不含真实 Key）
├── .gitignore # 排除 .env、缓存等文件
└── README.md


> 注意：Notebook 中 `KNOWLEDGE_FILE = "../knowledge.txt"`，即默认知识库文件位于 Notebook 的**上一级目录**。如果你把 `knowledge.txt` 和 Notebook 放在同一目录，请把该变量改为 `"knowledge.txt"`。

## 环境要求

- Python 3.9+
- 本地可访问的 Milvus 服务（默认 `http://localhost:19530`）
- SiliconFlow 账号（用于 BGE-M3 嵌入模型）
- CloseAI（或兼容 OpenAI 接口的服务）账号（用于对话模型）

### 安装依赖

```bash
pip install pymilvus langchain langchain-community langchain-text-splitters python-dotenv

```

如果项目提供了 requirements.txt，也可以：
```bash
pip install -r requirements.txt

```

启动 Milvus
本案例默认连接本地 Milvus。可使用 Docker 启动一个单机版 Milvus（示例）：
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest standalone
```
确保 http://localhost:19530 可访问，否则 Notebook 第 2 节初始化客户端会失败。


配置环境变量
复制模板并填入你自己的 Key：

```bash
cp .env.example .env
```

.env 中需要包含：


```dotenv
SILICONFLOW_API_KEY=你的 SiliconFlow API Key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

CLOSEAI_API_KEY=你的对话模型 API Key
CLOSEAI_BASE_URL=你的对话模型 Base URL
```
.env 已被 .gitignore 排除，请勿将真实 Key 提交到仓库。



运行方式
1、启动 Jupyter：

```bash
jupyter notebook
```
2、打开 Atguigu Assistant客服知识库.ipynb，按顺序执行：

    第 1 节 全局配置：确认 Milvus 地址、数据库名、集合名、知识库路径、嵌入模型与维度

    第 2 节 初始化 Milvus：创建数据库与 collection（若已存在会先删除重建）

    第 3 节 初始化 Embedding 模型：加载 BGE-M3

    第 4 节 读取文档并切分：加载 knowledge.txt 并切分为若干 chunk

    第 5 节 生成向量并写入 Milvus：向量化并 upsert 到集合

    第 6 节 创建 Agent：初始化对话模型与 agent

    第 7 节 检索：定义 retrieve() 函数

    第 8 节 生成与回答：调用 generate_answer() 完成一次完整问答

3、修改最后一节的问题即可测试不同 query：

```code
q = "为什么我在 7 天内申请退款，还是被拒了？"
generate_answer(q)
```


## 关键参数说明

    参数	                              说明
    MILVUS_URI	                Milvus 连接地址，默认 http://localhost:19530
    DB_NAME	                    自定义数据库名，默认 rag_tutorial
    COLLECTION_NAME	            向量集合名，默认 docs
    KNOWLEDGE_FILE	            知识库文件路径，默认 ../knowledge.txt
    EMBED_MODEL_NAME	        嵌入模型，默认 Pro/BAAI/bge-m3
    EMBED_DIM	                向量维度，BGE-M3 固定为 1024
    chunk_size / chunk_overlap	切分粒度，默认 200 / 80
    limit	                    检索返回条数，generate_answer 中默认 5

## 常见问题

    Q：运行第 2 节报连接错误？
    A：确认 Milvus 已启动且 MILVUS_URI 正确，端口 19530 可访问。
    
    Q：找不到 knowledge.txt？
    A：检查 KNOWLEDGE_FILE 路径。Notebook 默认读取上一级目录，若文件在同一目录请改为 "knowledge.txt"。
    
    Q：检索结果不相关或回答"我不知道"？
    A：检查 generate_answer 中是否调用了 retrieve(query, limit=5)（而非 retrieve(str, ...)），并确认知识库已成功写入 Milvus（第 5 节应显示 upsert_count 与记录数一致）。
    
    Q：嵌入或对话模型调用失败？
    A：检查 .env 中的 Key 与 Base URL 是否正确，以及对应服务是否可用。

# **免责声明**
本案例仅用于个人练习，知识库内容为虚构的客服问答示例，不构成任何真实产品承诺。

```text
几个小提醒：

- 如果你后面补了 `requirements.txt`，README 里"安装依赖"那段就不用改，已经兼容两种方式。
- 项目结构里我按"`knowledge.txt` 在根目录、Notebook 里用 `../knowledge.txt`"写的。如果你实际是把 `knowledge.txt` 放在 Notebook 同级目录，记得同步改 README 的项目结构和路径说明，避免别人照着跑不通。
- "常见问题"里我特意写了 `retrieve(str, ...)` 那个坑，既是排错指南，也提醒贡献者别踩回去。
```
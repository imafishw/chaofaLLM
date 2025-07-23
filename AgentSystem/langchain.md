# LangChain LLM工具实现 
> LangChain本身仍然坚守作为“模型能力增强器”的功能定位，并且逐渐稳定更新节奏和频率，虽说实际使用LangChain进行开发的代码量仍然没变，但模块划分更加清晰、功能更加丰富和稳定，逐步达到企业级应用水准。目前最新版LangChain的核心功能如下：

| 模块类别 | 示例功能 |
|---------|---------|
| 模型接口封装 | OpenAI、Claude、Cohere、Qwen 等模型统一调用方式 |
| 输出结构化 | 自动从模型中解析 JSON、Schema、函数签名、文档等 |
| Memory 管理 | Buffer、Summary、Entity、Conversation Memory 等 |
| Tool 接入 | Web 搜索、SQL 数据库、Python 执行器、API 代理等 |
| Agent 架构 | ReAct、Self-Ask、OpenAI Function Agent 等调度机制 |
| RAG 集成 | 多种 Retriever、Vector Store、文档拆分策略 |
| Server/API 发布 | 快速将链部署为 Web 服务或 A2A Agent |
| Debug & Callback | Token 使用统计、LangSmith 可视化追踪等 |

**与2023年下半年开源LangGraph，LangGraph作为基于LangChain的更高层次封装，能够更加便捷的搭建图结构的大模型工作流，也就是现在所谓的Multi-Agent系统，而LangGraph也是目前LangChain家族最核心的Multi-Agent开发框架。同时可以搭配LangGraph-Studio进行实时效果监测，实际效果如下所示**

![alt text](img/langchain.png)
## 一、模型输入输出

## 二、Retrieval

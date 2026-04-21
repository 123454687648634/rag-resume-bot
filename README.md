\# 简历智能问答机器人 (RAG Resume Bot)



\## 项目简介

基于 LangChain + Chroma + DeepSeek 的本地 RAG 应用。上传简历 PDF，即可与AI进行问答，了解求职者的技能、项目经验等信息。



\## 功能特性

\- \*\*查询改写\*\*：将用户口语问题优化为精准的检索词。

\- \*\*混合检索\*\*：结合向量检索（语义）与BM25（关键词），提升召回精度。

\- \*\*智能路由\*\*：自动判断用户意图，闲聊直接回复，专业问题走RAG流程。

\- \*\*流式对话\*\*：提供类似ChatGPT的逐字输出体验。



\## 技术栈

\- Python 3.11+

\- Streamlit (Web 界面)

\- LangChain (应用框架)

\- ChromaDB (向量存储)

\- DeepSeek API (大语言模型)

\- sentence-transformers (本地 Embedding)



\## 如何运行

1\. 克隆仓库：`git clone <你的仓库地址>`

2\. 安装依赖：`pip install -r requirements.txt`

3\. 替换 `web\_rag.py` 中的 `OPENAI\_API\_KEY` 为你自己的 DeepSeek API Key。

4\. 运行应用：`streamlit run web\_rag.py`



\## 在线演示

https://rag-resume-bot-nuebufox6mssgqzykj95bz.streamlit.app/


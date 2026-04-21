import streamlit as st
import os
import tempfile
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 设置页面标题和布局
st.set_page_config(page_title="📄 PDF 智能问答机器人", layout="wide")
st.markdown("""
<style>
    .stChatMessage { border-radius: 15px; }
    .stButton button { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)
st.title("📄 你的私人 RAG 助手")

# 从 Streamlit Cloud 的 Secrets 管理中读取密钥
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]

# 2. 初始化 Embeddings 和 LLM（缓存起来，避免重复加载）
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="deepseek-chat", temperature=0, streaming=True)

embeddings = load_embeddings()
llm = load_llm()

# 3. 混合检索函数（手动实现）
def hybrid_retrieve(query, vectorstore, documents, k=3):
    # 1. 向量检索
    vector_results = vectorstore.similarity_search(query, k=k)
    
    # 2. BM25 关键词检索
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    bm25_results = bm25_retriever.invoke(query)
    
    # 3. 合并去重（基于文档内容）
    seen_content = set()
    combined_results = []
    
    for doc in vector_results + bm25_results:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            combined_results.append(doc)
    
    # 调试：打印检索到的内容
    st.sidebar.write(f"🔍 向量检索结果数：{len(vector_results)}，BM25结果数：{len(bm25_results)}，合并后：{len(combined_results)}")
    if combined_results:
        st.sidebar.write(f"📝 第一条内容预览：{combined_results[0].page_content[:100]}")

    return combined_results[:k]

# 4. 处理 PDF 并构建向量库的函数
@st.cache_resource
def build_vectorstore(uploaded_file):
    # 将上传的文件保存为临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 加载 PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    st.sidebar.write(f"📄 解析到的文档长度：{len(documents)} 页，第一页前100字：{documents[0].page_content[:100] if documents else '空'}")

    # 切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # 构建向量库
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None
    )
    return vectorstore, documents, len(chunks)

# 5. 侧边栏：上传 PDF
with st.sidebar:
    st.header("📂 上传你的 PDF 文档")
    uploaded_file = st.file_uploader("选择一个 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🚀 处理文档"):
            with st.spinner("正在解析 PDF 并构建知识库，请稍候..."):
                try:
                    vectorstore, documents, chunk_count = build_vectorstore(uploaded_file)
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["documents"] = documents
                    st.success(f"✅ 文档处理完成！共切分为 {chunk_count} 个片段。")
                except Exception as e:
                    st.error(f"❌ 处理失败：{e}")

# 6. 初始化聊天历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 7. 显示聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 8. 用户输入框
if prompt := st.chat_input("请输入你的问题（基于上传的 PDF 内容）"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检查是否已经上传并处理了文档
    if "vectorstore" not in st.session_state:
        with st.chat_message("assistant"):
            st.markdown("⚠️ 请先在左侧上传并处理 PDF 文档。")
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ 请先在左侧上传并处理 PDF 文档。"})
    else:
 # ========== 新增：判断是否需要检索 ==========
        with st.spinner("🤔 分析问题意图..."):
            classify_prompt = ChatPromptTemplate.from_template(
                "判断用户的问题是否需要从已上传的PDF文档中查找答案。\n"
                "需要检索的情况：询问文档中的具体信息（如人名、技术、项目）、总结文档内容、回答基于文档的问题。\n"
                "不需要检索的情况：打招呼（你好）、闲聊、询问天气/时间等与文档无关的问题。\n"
                "只回答一个词：'是' 或 '否'，不要解释。\n"
                "用户问题：{query}"
            )
            classify_chain = classify_prompt | llm
            classify_response = classify_chain.invoke({"query": prompt})
            need_retrieval = classify_response.content.strip().lower() == "是"
            
            if not need_retrieval:
                # 直接让 LLM 回答，不检索 PDF
                with st.chat_message("assistant"):
                    with st.spinner("💬 回复中..."):
                        # 直接调用 LLM，不用检索到的上下文
                        direct_response = llm.invoke(prompt)
                        direct_answer = direct_response.content
                    st.markdown(direct_answer)
                # 保存回答到历史
                st.session_state.messages.append({"role": "assistant", "content": direct_answer})
                st.stop()  # 终止执行，跳过后续的检索和 RAG 流程
        # ===========================================


        # ========== 查询改写 ==========
        with st.spinner("🔄 正在优化检索词..."):
            rewrite_prompt_template = ChatPromptTemplate.from_template(
                "你是一个搜索专家。请将用户的口语化问题，改写成一个更精准、适合语义检索的短句。"
                "保留所有关键信息，去除礼貌用语和无关词汇。只返回改写后的句子，不要解释。\n"
                "用户问题：{query}"
            )
            rewrite_chain = rewrite_prompt_template | llm
            rewrite_response = rewrite_chain.invoke({"query": prompt})
            rewritten_query = rewrite_response.content.strip()
            st.caption(f"🔍 优化后的检索词：{rewritten_query}")
        
        # ========== 混合检索 ==========
        retrieved_docs = hybrid_retrieve(
            query=rewritten_query,
            vectorstore=st.session_state.vectorstore,
            documents=st.session_state.documents,
            k=3
        )
        
        # 生成回答
        question_answer_chain = create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_template(
                "请根据以下上下文内容回答问题。如果无法从上下文中找到答案，请直接说'不知道'。\n\n"
                "上下文：\n{context}\n\n"
                "问题：{input}\n\n"
                "回答："
            )
        )
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 思考中..."):
                # 直接传入文档对象列表，而不是手动拼接的字符串
                answer = question_answer_chain.invoke({
                    "context": retrieved_docs,  # 注意：这里是文档列表，不是字符串
                    "input": rewritten_query
                })
            st.markdown(answer)
        
        # 保存回答到历史
        st.session_state.messages.append({"role": "assistant", "content": answer})
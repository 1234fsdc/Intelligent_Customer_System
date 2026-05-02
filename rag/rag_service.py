"""
RAG（检索增强生成）服务模块

【什么是RAG】
RAG = Retrieval-Augmented Generation（检索增强生成）
传统LLM的问题：知识来自训练数据，可能过时或 hallucination（胡说八道）
RAG的解决方案：先从外部知识库检索相关资料，再让LLM基于资料回答

【RAG工作流程】
1. 用户提问
2. 将问题向量化，在向量库中检索相关文档
3. 将检索到的文档和用户问题组合成Prompt
4. LLM基于参考资料生成回答
5. 返回回答给用户

【为什么用RAG】
1. 准确性：回答基于真实资料，减少幻觉
2. 可更新：只需更新知识库，无需重新训练模型
3. 可追溯：可以展示回答的参考来源
4. 成本低：比微调模型便宜得多
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from model.factory import chat_model
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService:
    """
    RAG摘要服务类
    
    【职责】
    1. 管理向量库检索器
    2. 构建RAG提示词模板
    3. 执行检索+生成的完整流程
    
    【为什么封装成类】
    将检索和生成逻辑封装，对外提供简洁的rag_summarize接口，
    调用者不需要知道内部如何实现检索和Prompt构建。
    """
    
    def __init__(self):
        """
        初始化RAG服务
        
        【初始化步骤】
        1. 创建向量存储服务实例（管理Chroma向量库）
        2. 获取检索器（用于语义检索）
        3. 加载RAG提示词模板（告诉AI如何基于资料回答）
        4. 构建LangChain Chain（提示词→模型→输出解析器）
        """
        # 【向量存储服务】
        # 封装了Chroma向量库的加载、检索等功能
        self.vector_store = VectorStoreService()
        
        # 【检索器】
        # 输入查询文本，返回相关文档列表
        self.retriever = self.vector_store.get_retriever()
        
        # 【提示词模板】
        # 从配置文件加载，格式如：
        # "基于以下参考资料回答问题：\n{context}\n\n用户问题：{input}"
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        
        # 【构建Chain】
        # Chain是LangChain的核心概念，将多个组件串联起来：
        # PromptTemplate → ChatModel → StrOutputParser
        # 输入：{"input": 用户问题, "context": 参考资料}
        # 输出：AI生成的回答字符串
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def rag_summarize(self, query: str) -> str:
        """
        执行RAG检索和摘要生成
        
        【执行流程】
        1. 调用retriever检索相关文档
        2. 将文档格式化为上下文文本
        3. 调用Chain生成最终回答
        
        【参数】
        query: 用户的查询问题
        
        【返回值】
        AI基于参考资料生成的回答字符串
        """
        # 【步骤1：检索相关文档】
        # retriever将query向量化，在Chroma中搜索最相似的文档
        # 返回TopK个Document对象（k由配置决定，默认3）
        context_docs = self.retriever.invoke(query)
        
        # 【步骤2：格式化上下文】
        # 将多个文档内容拼接成一个字符串，添加编号便于AI理解
        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += f"【参考资料{i}】: {doc.page_content}\n"
        
        # 【步骤3：生成回答】
        # 将用户问题和参考资料一起传给LLM，生成最终回答
        return self.chain.invoke({"input": query, "context": context})


if __name__ == '__main__':
    # 【模块自测】
    # 测试RAG服务是否正常工作
    service = RagSummarizeService()
    result = service.rag_summarize("如何清洁扫地机器人？")
    print(result)

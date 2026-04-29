"""
RAG检索增强生成服务模块

【什么是RAG】
RAG = Retrieval-Augmented Generation（检索增强生成）
核心思想：先检索相关知识，再让AI基于知识回答问题。
解决大模型的"幻觉"问题（胡说八道），让回答有依据、可追溯。

【RAG工作流程】
1. 用户提问 → 2. 向量检索相关文档 → 3. 将文档+问题一起输入AI → 4. AI基于文档回答

【为什么用RAG】
- 大模型训练数据有截止日期，不知道最新信息
- 大模型不知道企业内部私有知识
- RAG让AI能"查资料"再回答，提高准确性和可信度
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


def print_prompt(prompt):
    """
    打印提示词的调试函数
    
    【为什么】开发调试时需要查看最终生成的提示词内容，确保格式正确
    【怎么做】打印分隔线后输出提示词字符串，最后返回prompt供链式调用继续传递
    """
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    """
    RAG总结服务类
    
    【职责】协调向量检索和文本生成两个环节，封装RAG完整流程
    【设计模式】外观模式（Facade），对外提供简洁的rag_summarize接口，隐藏内部复杂逻辑
    """
    
    def __init__(self):
        """
        初始化RAG服务
        
        【初始化步骤】
        1. 创建向量存储服务（连接Chroma数据库）
        2. 获取检索器（用于搜索相似文档）
        3. 加载RAG提示词模板（告诉AI如何基于参考资料回答）
        4. 创建PromptTemplate对象（用于格式化提示词）
        5. 获取大模型实例（通义千问）
        6. 构建处理链（Chain），将各环节串联
        """
        # 【为什么用VectorStoreService】
        # 封装了Chroma向量库的操作，提供文档存储、检索等能力
        self.vector_store = VectorStoreService()
        
        # 【什么是Retriever】
        # 检索器，接收查询字符串，返回最相似的文档列表
        # 内部使用向量相似度计算（余弦相似度）找到相关文档
        self.retriever = self.vector_store.get_retriever()
        
        # 【为什么加载提示词】
        # RAG需要特殊的提示词模板，告诉AI："基于以下参考资料回答问题"
        self.prompt_text = load_rag_prompts()
        
        # 【PromptTemplate的作用】
        # 将提示词模板中的变量（如{input}、{context}）替换为实际值
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        
        # 【为什么保存model引用】
        # 用于构建处理链，实际调用大模型生成回答
        self.model = chat_model
        
        # 【为什么初始化chain】
        # LangChain的链式调用（|运算符）将多个组件串联，数据从左到右流动
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化LangChain处理链
        
        【什么是Chain】
        Chain是LangChain的核心概念，将多个处理步骤串联，数据流式传递。
        用|运算符连接，类似Unix管道的概念。
        
        【本项目的Chain流程】
        prompt_template | print_prompt | model | StrOutputParser
        
        1. prompt_template: 接收{"input": ..., "context": ...}，填充模板生成完整提示词
        2. print_prompt: 打印提示词（调试用），原样返回
        3. model: 将提示词传给大模型，生成回答
        4. StrOutputParser: 解析模型输出，提取纯文本内容
        
        【为什么用StrOutputParser】
        # 大模型返回的是复杂的Message对象，Parser将其转换为普通字符串
        """
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        检索相关文档
        
        【参数】
        query: 用户查询字符串（如"小户型适合哪些扫地机器人"）
        
        【返回值】
        Document对象列表，每个Document包含：
        - page_content: 文档内容（字符串）
        - metadata: 元数据（如来源文件、页码等）
        
        【检索原理】
        1. 将query转为向量（embedding）
        2. 在Chroma向量库中查找最相似的向量
        3. 返回对应的原始文档
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        RAG总结主方法
        
        【完整流程】
        1. 根据query检索相关文档
        2. 将文档格式化为上下文字符串
        3. 调用chain生成回答
        
        【参数】
        query: 用户问题
        
        【返回值】
        AI生成的回答字符串（基于检索到的参考资料）
        """
        # 【步骤1】检索相关文档
        context_docs = self.retriever_docs(query)

        # 【步骤2】格式化文档为上下文字符串
        # 【为什么拼接成字符串】
        # 大模型只能处理文本，需要将Document对象转为字符串格式
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            # 【格式化格式】
            # 【参考资料N】: 参考资料：内容 | 参考元数据：元数据
            # 清晰的格式帮助AI理解哪些是可参考的内容
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        # 【步骤3】调用处理链生成回答
        # 【invoke参数】
        # 字典的key必须和prompt_template中的变量名匹配
        # {input}对应用户问题，{context}对应参考资料
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    # 【模块自测】
    # 创建RAG服务实例，测试小户型扫地机器人问题
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))

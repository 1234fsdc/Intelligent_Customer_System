"""
AI工具集模块

【什么是工具（Tools）】
工具是AI可以调用的外部功能，扩展了AI的能力边界。
例如：查知识库、查天气、查用户数据等。

【为什么用@tool装饰器】
LangChain的@tool装饰器将普通Python函数转换为AI可调用的工具，
自动解析函数签名、类型注解和docstring，生成工具描述。
"""

from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService

# 【全局RAG服务实例】
# 初始化一次，避免重复加载向量库
rag = RagSummarizeService()


@tool(description="从向量存储中检索参考资料，用于回答用户关于产品知识、使用指南、故障排除等问题。入参为query（检索词）")
def rag_summarize(query: str) -> str:
    """
    RAG知识库检索工具
    
    【功能】
    从Chroma向量库中检索与用户问题相关的文档资料，
    用于辅助AI生成准确、专业的回答。
    
    【参数】
    query: 检索词，通常是用户的问题或关键词
    
    【返回值】
    检索到的参考资料文本，包含相关文档内容
    
    【使用场景】
    - 用户咨询产品使用方法
    - 用户询问故障排除方案
    - 用户需要了解产品功能
    - 任何需要专业知识回答的场景
    """
    return rag.rag_summarize(query)

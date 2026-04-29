"""
模型工厂模块

【为什么用工厂模式】
工厂模式是一种创建型设计模式，将对象的创建逻辑封装在工厂类中。
好处：
1. 解耦：调用者不需要知道具体如何创建对象
2. 可扩展：切换模型时只需修改工厂，不影响调用方
3. 统一管理：集中管理模型配置和初始化

【本项目使用抽象工厂模式】
定义抽象基类BaseModelFactory，所有具体工厂都继承它并实现generator方法。
这样可以用统一的方式创建不同类型的模型（聊天模型、嵌入模型）。
"""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    """
    模型工厂抽象基类
    
    【ABC和@abstractmethod的作用】
    ABC（Abstract Base Class）是Python的抽象基类，不能直接被实例化。
    @abstractmethod标记抽象方法，子类必须实现这个方法，否则无法实例化。
    
    【为什么用抽象基类】
    强制所有具体工厂遵循相同的接口规范，确保它们都有generator方法。
    """
    
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """
        生成模型实例的抽象方法
        
        【返回值】
        Optional[Embeddings | BaseChatModel]：
        - Embeddings：文本嵌入模型，用于将文本转为向量
        - BaseChatModel：聊天模型，用于对话生成
        - Optional表示可能返回None（如果创建失败）
        
        【为什么返回Union类型】
        不同类型的工厂返回不同类型的模型，用Union统一表示。
        """
        pass


class ChatModelFactory(BaseModelFactory):
    """
    聊天模型工厂
    
    【职责】创建大语言模型实例（通义千问）
    
    【为什么用ChatTongyi】
    ChatTongyi是LangChain社区提供的通义千问封装，兼容OpenAI接口格式，
    支持流式输出、函数调用等高级功能。
    """
    
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """
        创建聊天模型实例
        
        【实现】
        从配置文件读取模型名称（如"qwen3-max"），传给ChatTongyi初始化。
        
        【配置化设计】
        模型名称从rag_conf读取，而不是硬编码，便于切换模型版本。
        """
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    """
    嵌入模型工厂
    
    【职责】创建文本嵌入模型实例
    
    【什么是嵌入模型】
    将文本转换为高维向量（如1536维），捕获语义信息。
    相似语义的文本向量距离更近，用于RAG检索。
    
    【为什么用DashScopeEmbeddings】
    DashScope是阿里云的AI模型服务，text-embedding-v4是高性能的中文嵌入模型，
    与通义千问同生态，兼容性好。
    """
    
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """
        创建嵌入模型实例
        
        【实现】
        从配置文件读取嵌入模型名称（如"text-embedding-v4"），
        传给DashScopeEmbeddings初始化。
        """
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


# 【为什么全局实例化】
# 模型初始化需要加载权重、建立连接，比较耗时。
# 全局实例化确保只创建一次，后续复用，提高性能。

# 【聊天模型实例】
# 用于AI对话生成，被ReactAgent、RagSummarizeService等模块使用
chat_model = ChatModelFactory().generator()

# 【嵌入模型实例】
# 用于文本向量化，被VectorStoreService使用
embed_model = EmbeddingsFactory().generator()

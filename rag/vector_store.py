"""
Chroma向量存储服务模块

【什么是向量数据库】
向量数据库专门用于存储和检索高维向量（嵌入向量）。
在RAG中，文档被转换为向量后存入向量数据库，检索时通过向量相似度匹配找到相关文档。

【为什么用Chroma】
Chroma是一个开源的轻量级向量数据库，特点：
1. 易于使用：API简洁，与LangChain集成良好
2. 持久化：支持将数据保存到磁盘，重启后不丢失
3. 本地运行：无需外部服务，适合开发和小型项目
4. 免费开源：无使用限制
"""

import os
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from utils.config_handler import chroma_conf
from model.factory import embed_model
from utils.logger_handler import logger


class VectorStoreService:
    """
    Chroma向量存储服务类
    
    【职责】
    1. 管理Chroma向量库实例（创建、加载）
    2. 文档加载、分片、向量化存储
    3. 提供检索器（retriever）用于语义检索
    
    【工作流程】
    文档文件 → 加载为Document对象 → 切分为小块 → 向量化 → 存入Chroma
    """
    
    def __init__(self):
        """
        初始化向量存储服务
        
        【Chroma参数说明】
        - collection_name: 集合名称，类似数据库中的表名
        - embedding_function: 嵌入模型，用于将文本转为向量
        - persist_directory: 持久化目录，向量数据保存到这里
        """
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # 从配置读取集合名
            embedding_function=embed_model,  # 使用全局嵌入模型实例
            persist_directory=chroma_conf["persist_directory"],  # 持久化目录
        )
        
        # 【为什么用RecursiveCharacterTextSplitter】
        # 递归字符文本分割器，按优先级尝试不同分隔符：
        # 1. 先按段落分隔（\n\n）
        # 2. 再按行分隔（\n）
        # 3. 再按句子分隔（。！？等）
        # 4. 最后按字符分隔
        # 这样能保持语义完整性，不会在句子中间切断
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],  # 每块大小（字符数）
            chunk_overlap=chroma_conf["chunk_overlap"],  # 块间重叠（字符数）
            separators=chroma_conf["separators"],  # 分隔符优先级列表
        )

    def add_documents_from_folder(self, folder_path: str):
        """
        从文件夹加载文档并添加到向量库
        
        【为什么遍历文件夹】
        知识库通常有多个文档（txt、pdf等），批量处理更高效。
        
        【参数】
        folder_path: 包含文档的文件夹路径
        """
        folder_path = os.path.abspath(folder_path)
        
        # 【为什么检查文件夹存在】
        # 避免路径错误导致程序崩溃
        if not os.path.exists(folder_path):
            logger.warning(f"[add_documents_from_folder]文件夹不存在: {folder_path}")
            return
        
        # 【遍历文件夹，处理每个文档文件】
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # 【根据文件扩展名选择加载器】
            # .txt文件用TextLoader，.pdf文件用PyPDFLoader
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                continue  # 跳过不支持的文件类型
            
            # 【加载文档为Document对象列表】
            # Document包含page_content（文本内容）和metadata（元数据）
            documents = loader.load()
            
            # 【文档分片】
            # 将长文档切分为多个小块，每个块大小由chunk_size控制
            # 重叠部分确保上下文不丢失
            texts = self.spliter.split_documents(documents)
            
            # 【添加到向量库】
            # 内部流程：文本 → 嵌入模型 → 向量 → 存入Chroma
            self.vector_store.add_documents(texts)
            logger.info(f"[add_documents_from_folder]已处理文件: {filename}")
        
        logger.info(f"[add_documents_from_folder]文件夹处理完成: {folder_path}")

    def get_retriever(self):
        """
        获取检索器
        
        【什么是retriever】
        retriever是LangChain封装的检索接口，输入查询文本，返回相关文档。
        内部流程：查询文本 → 向量化 → 在Chroma中搜索相似向量 → 返回TopK文档
        
        【返回值】
        VectorStoreRetriever对象，可调用invoke(query)进行检索
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["k"]}  # 返回最相关的k个文档
        )


if __name__ == '__main__':
    # 【模块自测】
    # 测试向量库初始化和文档加载
    service = VectorStoreService()
    service.add_documents_from_folder(chroma_conf["data_path"])
    print("向量库初始化完成")

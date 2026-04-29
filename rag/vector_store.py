"""
向量存储服务模块

【什么是向量存储】
将文本转换为高维向量（embedding）并存储，支持基于语义相似度的检索。
相似语义的文本在向量空间中距离更近，即使关键词不同也能找到相关内容。

【核心技术】
- Embedding（嵌入）：将文本转为数值向量，捕获语义信息
- 向量相似度：用余弦相似度等算法计算向量间的距离
- ANN（近似最近邻）：高效检索最相似的向量（如HNSW算法）

【为什么用Chroma】
Chroma是开源的向量数据库，轻量易用，支持持久化存储，与LangChain集成良好。
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    """
    向量存储服务类
    
    【职责】
    1. 管理Chroma向量数据库的连接和操作
    2. 提供文档加载、分片、入库的完整流程
    3. 提供检索器（Retriever）供RAG服务使用
    
    【设计思路】
    封装底层向量库的复杂性，对外提供简洁的load_document和get_retriever接口。
    """
    
    def __init__(self):
        """
        初始化向量存储服务
        
        【Chroma初始化参数】
        - collection_name: 集合名称，类似数据库的表名，用于区分不同知识库
        - embedding_function: 嵌入函数，将文本转为向量的模型
        - persist_directory: 持久化目录，向量数据保存到磁盘的路径
        
        【为什么持久化】
        如果不持久化，每次重启应用向量数据都会丢失，需要重新加载文档。
        持久化后，启动时直接读取已有数据，加快启动速度。
        """
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # 从配置文件读取集合名
            embedding_function=embed_model,  # 使用DashScope的embedding模型
            persist_directory=chroma_conf["persist_directory"],  # 向量库保存路径
        )

        # 【为什么用RecursiveCharacterTextSplitter】
        # 长文档不能直接存入向量库（超出模型上下文限制），需要切分成小块（chunk）。
        # RecursiveCharacterTextSplitter是LangChain推荐的分片器，它会按优先级尝试不同的分隔符，
        # 尽量在语义完整的位置切分（如段落、句子），避免切断句子。
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],      # 每个分片的最大字符数（200字符）
            chunk_overlap=chroma_conf["chunk_overlap"], # 相邻分片的重叠字符数（20字符）
            separators=chroma_conf["separators"],       # 分隔符优先级列表
            length_function=len,                        # 计算长度的函数
        )

    def get_retriever(self):
        """
        获取检索器
        
        【什么是Retriever】
        检索器是一个可调用对象，接收查询字符串，返回最相似的文档列表。
        内部自动完成：查询→向量化→相似度计算→返回TopK结果。
        
        【search_kwargs={"k": 3}】
        指定返回最相似的3个文档。k值越大，召回率越高但精度可能下降。
        """
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        加载文档到向量库
        
        【完整流程】
        1. 扫描数据文件夹，获取所有允许类型的文件
        2. 计算每个文件的MD5，检查是否已处理过（去重）
        3. 读取文件内容，转为Document对象
        4. 将长文档分片成小块
        5. 存入Chroma向量库
        6. 记录MD5，避免下次重复加载
        
        【为什么用MD5去重】
        文件内容不变时MD5值相同，通过比对MD5可以识别已加载的文件，
        避免重复入库浪费时间和存储空间。
        """

        def check_md5_hex(md5_for_check: str):
            """
            检查MD5是否已处理过
            
            【实现】读取md5_hex_store文件，逐行比对。
            如果文件不存在，创建空文件并返回False（未处理）。
            """
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 【为什么创建空文件】
                # 首次运行时文件不存在，创建空文件避免后续报错
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            """
            保存MD5到记录文件
            
            【实现】以追加模式（"a"）打开文件，写入MD5值和换行符。
            追加模式不会覆盖已有内容，只会添加到文件末尾。
            """
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """
            根据文件类型选择对应的加载器
            
            【为什么不同文件用不同加载器】
            PDF和TXT的格式不同，需要专门的解析器。
            PDF可能有多页结构，TXT是纯文本。
            """
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        # 【获取待处理文件列表】
        # 扫描data目录，返回所有允许类型（txt、pdf）的文件路径
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 【遍历处理每个文件】
        for path in allowed_files_path:
            # 获取文件的MD5值
            md5_hex = get_file_md5_hex(path)

            # 【去重检查】如果MD5已存在，跳过此文件
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                # 【步骤1】加载文件内容为Document对象列表
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # 【步骤2】文档分片
                # 【为什么分片】长文档超出embedding模型上下文限制，且细粒度检索更精准
                # 【overlap的作用】相邻分片有重叠，避免关键信息被切分在两个分片中丢失
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 【步骤3】存入向量库
                # 【内部流程】Chroma会自动调用embedding_function将文本转为向量并存储
                self.vector_store.add_documents(split_document)

                # 【步骤4】记录MD5，标记为已处理
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # 【exc_info=True】记录完整的异常堆栈，便于调试定位问题
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    # 【模块自测】
    # 创建向量存储服务，加载文档，测试检索功能
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    # 测试查询"迷路"，查看返回的相关文档片段
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)

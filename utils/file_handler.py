"""
文件处理工具模块

【职责】
提供文件相关的工具函数，包括：
1. 文件MD5计算（用于去重校验）
2. 目录扫描（获取指定类型的文件列表）
3. 文档加载（PDF、TXT等格式）

【设计思路】
将文件操作封装成独立函数，便于复用和统一错误处理。
"""

import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str) -> str | None:
    """
    计算文件的MD5哈希值（十六进制字符串）
    
    【什么是MD5】
    MD5（Message-Digest Algorithm 5）是一种广泛使用的哈希算法，
    将任意长度的数据映射为128位（16字节）的哈希值，通常表示为32位十六进制字符串。
    
    【为什么用MD5】
    1. 唯一性：不同内容的文件几乎不可能有相同的MD5值
    2. 快速：计算速度快，适合大文件
    3. 确定性：相同内容的文件MD5值一定相同
    
    【应用场景】
    - 文件去重：比较MD5值判断文件是否已处理过
    - 完整性校验：下载后计算MD5验证文件是否损坏
    
    【参数】
    filepath: 文件路径
    
    【返回值】
    MD5十六进制字符串，计算失败返回None
    
    【为什么分块读取】
    大文件一次性读取会占用大量内存，分块读取（4KB/次）可以处理任意大小的文件。
    """
    # 【前置校验】检查文件是否存在
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    # 【前置校验】检查路径是否为文件（而非目录）
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    # 【创建MD5对象】
    md5_obj = hashlib.md5()

    chunk_size = 4096  # 4KB分块大小，平衡内存占用和IO次数
    
    try:
        # 【二进制模式打开文件】
        # "rb" = read binary，MD5计算需要原始字节，不能经过文本编码转换
        with open(filepath, "rb") as f:
            # 【循环读取分块】
            # 使用海象运算符:=，在while条件中赋值并判断
            # f.read(chunk_size)读取4KB数据，返回空字节串时（文件读完）循环结束
            while chunk := f.read(chunk_size):
                # 【更新MD5对象】
                # 将读取的数据块喂给MD5对象，内部更新哈希状态
                md5_obj.update(chunk)

            # 【获取十六进制哈希值】
            # hexdigest()将128位二进制哈希转为32位十六进制字符串
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        # 【异常处理】
        # 记录错误日志，返回None表示计算失败
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]) -> tuple[str]:
    """
    获取目录中指定类型的文件列表
    
    【功能】
    扫描指定目录，返回所有符合后缀名要求的文件路径
    
    【参数】
    path: 要扫描的目录路径
    allowed_types: 允许的文件后缀名元组（如(".txt", ".pdf")）
    
    【返回值】
    文件路径元组，每个路径都是绝对路径
    
    【为什么返回tuple而不是list】
    tuple是不可变类型，作为返回值更安全，调用方无法意外修改
    """
    files = []

    # 【前置校验】检查路径是否为目录
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types  # 【注意】这里返回allowed_types可能有bug，应该返回空元组

    # 【遍历目录】
    # os.listdir(path)返回目录下所有文件和文件夹的名称列表
    for f in os.listdir(path):
        # 【后缀名检查】
        # str.endswith()支持元组参数，检查字符串是否以任一后缀结尾
        if f.endswith(allowed_types):
            # 【拼接完整路径】
            # os.path.join根据操作系统使用正确的路径分隔符（Windows用\，Linux/Mac用/）
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd: str = None) -> list[Document]:
    """
    加载PDF文件
    
    【功能】
    使用LangChain的PyPDFLoader解析PDF文件，提取文本内容
    
    【参数】
    filepath: PDF文件路径
    passwd: PDF密码（如果是加密PDF），默认None
    
    【返回值】
    Document对象列表，每个Document包含一页的内容和元数据
    
    【Document结构】
    - page_content: 提取的文本内容
    - metadata: 字典，包含source（文件路径）、page（页码）等信息
    
    【为什么用PyPDFLoader】
    LangChain封装的PDF加载器，自动处理多页PDF，每页生成一个Document
    """
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    """
    加载TXT文本文件
    
    【功能】
    使用LangChain的TextLoader加载纯文本文件
    
    【参数】
    filepath: 文本文件路径
    
    【返回值】
    Document对象列表（通常只有一个Document，包含整个文件内容）
    
    【编码说明】
    使用UTF-8编码读取，支持中文和其他多语言文本
    
    【为什么用TextLoader而不是直接open】
    TextLoader统一封装了文件读取和Document对象创建，与PDF等其他加载器接口一致
    """
    return TextLoader(filepath, encoding="utf-8").load()

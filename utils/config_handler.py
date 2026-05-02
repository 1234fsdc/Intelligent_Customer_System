"""
配置管理模块

【为什么用YAML配置文件】
YAML（YAML Ain't Markup Language）是一种人类可读的数据序列化格式。
优点：
1. 可读性好：缩进表示层级，比JSON更易读
2. 支持注释：可以在配置中添加说明
3. 类型丰富：支持字符串、数字、布尔值、列表、字典等
4. 无额外符号：不需要引号和逗号，简洁

【配置分离的好处】
1. 环境适配：不同环境（开发/测试/生产）用不同配置
2. 动态调整：修改配置无需改代码，重启即可生效
3. 权限管理：敏感配置（如API密钥）可以单独管理
"""

import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    """
    加载RAG配置
    
    【配置内容】
    - chat_model_name: 聊天模型名称（如"qwen3-max"）
    - embedding_model_name: 嵌入模型名称（如"text-embedding-v4"）
    
    【参数】
    config_path: 配置文件路径，默认"config/rag.yml"
    encoding: 文件编码，默认UTF-8（支持中文）
    
    【返回值】
    字典对象，包含配置项的键值对
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    """
    加载Chroma向量库配置
    
    【配置内容】
    - collection_name: 集合名称
    - persist_directory: 持久化目录
    - k: 检索返回的文档数量
    - data_path: 知识库数据文件夹
    - chunk_size: 文档分片大小
    - chunk_overlap: 分片重叠大小
    - separators: 分片分隔符列表
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    """
    加载提示词配置
    
    【配置内容】
    - main_prompt: 系统提示词（定义AI角色和能力）
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    """
    加载Agent配置
    
    【配置内容】
    - device_data_dir: 设备数据目录
    - 其他Agent运行参数
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 【为什么全局加载配置】
# 配置在应用生命周期内是不变的，全局加载后各处直接引用，避免重复读取文件。
# 模块被导入时执行，确保其他模块导入时配置已准备好。

# RAG相关配置（模型名称等）
rag_conf = load_rag_config()

# Chroma向量库配置
chroma_conf = load_chroma_config()

# 提示词配置
prompts_conf = load_prompts_config()

# Agent配置
agent_conf = load_agent_config()


if __name__ == '__main__':
    # 【模块自测】
    # 测试配置是否正确加载
    print(rag_conf)
    print(chroma_conf)
    print(prompts_conf)
    print(agent_conf)

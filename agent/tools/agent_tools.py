"""
智能体工具集模块

【什么是工具（Tools）】
工具是AI可调用的外部功能，让AI能够：
- 获取实时信息（如天气、时间）
- 查询私有数据（如知识库、用户记录）
- 执行特定操作（如生成报告）

【为什么用@tool装饰器】
LangChain的@tool装饰器将普通函数包装成AI可调用的工具：
1. 提取函数的docstring作为工具描述，告诉AI这个工具是做什么的
2. 提取函数的参数类型和名称，让AI知道如何传参
3. 统一接口格式，便于agent管理和调用

【工具调用流程】
1. AI分析用户问题，判断是否需要工具
2. AI输出JSON格式的工具调用请求（包含工具名和参数）
3. LangChain解析JSON，找到对应的函数并执行
4. 函数执行结果返回给AI，AI基于结果生成最终回答
"""

import os
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

# 【为什么全局初始化RAG服务】
# RAG服务初始化需要加载模型和向量库，比较耗时。
# 全局初始化避免每次调用工具时都重新创建，提高性能。
rag = RagSummarizeService()

# 【模拟数据】
# 这些数组用于模拟用户ID和月份数据，实际项目中应从数据库或外部系统获取
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

# 【外部数据缓存】
# 用于缓存从CSV文件加载的用户使用记录数据，避免重复读取文件
external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """
    RAG知识库检索工具
    
    【功能】根据用户查询，从向量知识库中检索相关资料并生成总结
    
    【参数】
    query: 用户的查询问题（如"扫地机器人如何保养"）
    
    【返回值】
    基于知识库资料生成的回答字符串
    
    【使用场景】
    用户询问产品相关问题（如选购指南、故障排除、维护保养）时调用
    """
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """
    天气查询工具
    
    【功能】获取指定城市的当前天气信息
    
    【参数】
    city: 城市名称（如"深圳"、"北京"）
    
    【返回值】
    格式化的天气信息字符串
    
    【注意】
    当前实现是模拟数据，实际项目应调用真实的天气API（如和风天气、OpenWeatherMap）
    """
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    用户定位工具
    
    【功能】获取用户当前所在的城市
    
    【返回值】
    城市名称字符串（如"深圳"、"合肥"、"杭州"）
    
    【注意】
    当前实现是随机返回，实际项目应通过IP定位或GPS获取真实位置
    """
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """
    用户ID获取工具
    
    【功能】获取当前用户的唯一标识ID
    
    【返回值】
    用户ID字符串（如"1001"）
    
    【使用场景】
    需要查询用户专属数据（如使用记录、个人报告）时使用
    
    【注意】
    当前实现是随机返回，实际项目应从登录会话中获取真实用户ID
    """
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """
    当前月份获取工具
    
    【功能】获取当前的年份-月份
    
    【返回值】
    月份字符串，格式为"YYYY-MM"（如"2025-06"）
    
    【使用场景】
    生成月度报告、查询某月数据时使用
    
    【注意】
    当前实现是随机返回，实际项目应使用datetime获取真实时间
    """
    return random.choice(month_arr)


def generate_external_data():
    """
    加载外部数据到内存
    
    【数据来源】
    从CSV文件读取用户使用记录，包含用户ID、月份、特征、效率、耗材、对比等字段
    
    【数据结构】
    external_data = {
        "user_id": {
            "month": {"特征": xxx, "效率": xxx, "耗材": xxx, "对比": xxx},
            ...
        },
        ...
    }
    
    【为什么用双层字典】
    第一层：用户ID → 该用户的所有月份数据
    第二层：月份 → 该月的详细指标
    这样可以通过external_data[user_id][month]快速查询特定用户在特定月份的数据
    
    【为什么缓存到内存】
    CSV文件读取是IO操作，比较慢。缓存到字典后，后续查询直接从内存获取，O(1)时间复杂度。
    """
    # 【为什么检查if not external_data】
    # 避免重复加载，如果已经加载过（字典非空），直接跳过
    if not external_data:
        # 从配置文件获取外部数据文件路径
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        # 【文件存在性检查】
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        # 【读取CSV文件】
        # [1:]跳过表头行，从第二行开始读取数据
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                # 【解析CSV行】
                # 按逗号分割，去除引号，得到各字段值
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                # 【构建嵌套字典结构】
                # 如果用户ID不存在，先创建空字典
                if user_id not in external_data:
                    external_data[user_id] = {}

                # 存储该用户该月份的数据
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    外部数据查询工具
    
    【功能】查询指定用户在指定月份的使用记录
    
    【参数】
    user_id: 用户ID（如"1001"）
    month: 月份（格式"YYYY-MM"，如"2025-06"）
    
    【返回值】
    该用户该月份的使用记录字符串，未找到返回空字符串
    
    【使用场景】
    生成用户使用报告时，获取历史数据作为参考
    """
    # 【确保数据已加载】
    generate_external_data()

    try:
        # 【查询数据】
        # 使用嵌套字典索引，如果key不存在会抛出KeyError
        return str(external_data[user_id][month])
    except KeyError:
        # 【异常处理】
        # 记录警告日志，返回空字符串表示未找到数据
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """
    报告上下文填充工具
    
    【功能】触发报告生成模式的上下文切换
    
    【为什么需要这个工具】
    这是一个特殊的"信号工具"，本身不执行实际功能，
    但调用后会触发中间件report_prompt_switch，将后续对话切换到"报告生成"模式。
    
    【使用场景】
    当用户说"给我生成报告"时，AI会先调用此工具，然后使用专门的报告生成提示词模板。
    
    【返回值】
    固定字符串，表示工具已调用
    """
    return "fill_context_for_report已调用"

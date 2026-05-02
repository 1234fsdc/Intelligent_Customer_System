"""
ReAct智能体核心模块 - 电子设备推荐与知识库问答系统

【什么是ReAct】
ReAct = Reasoning（推理）+ Acting（行动），是一种让AI能够"思考→行动→观察→再思考"的循环架构。
AI不仅能回答问题，还能调用工具（如设备检索、知识库查询）获取信息，然后基于工具结果继续推理。

【为什么用LangChain的create_agent】
LangChain封装了ReAct的复杂逻辑，包括：
- 提示词模板管理（告诉AI如何思考、何时调用工具）
- 工具调用解析（从AI输出中提取工具名称和参数）
- 执行循环管理（自动处理"思考→调用工具→观察结果→再思考"的循环）
"""

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.device_tools import search_devices, filter_by_budget, match_by_scene, sort_by_value
from agent.tools.agent_tools import rag_summarize


class ReactAgent:
    """
    ReAct智能体封装类 - 电子设备推荐与知识库问答系统
    
    【为什么封装成类】
    1. 统一管理agent实例的生命周期（创建、复用）
    2. 隐藏LangChain的复杂细节，对外提供简洁的execute_stream接口
    3. 便于在Streamlit的session_state中存储和复用
    """
    
    def __init__(self):
        """
        初始化ReAct智能体
        
        【create_agent参数说明】
        - model: 大语言模型实例（这里是通义千问），负责理解问题和生成回答
        - system_prompt: 系统提示词，告诉AI它的角色和能力边界
        - tools: 工具列表，AI可以调用的外部功能（如设备检索、知识库查询）
        
        【工具是如何被AI调用的】
        1. AI分析用户问题，判断是否需要工具
        2. 如果需要，AI会输出特定格式的JSON，包含工具名和参数
        3. LangChain解析这个JSON，调用对应的Python函数
        4. 函数执行结果返回给AI，AI基于结果继续回答
        """
        self.agent = create_agent(
            model=chat_model,  # 从工厂获取的通义千问模型实例
            system_prompt=load_system_prompts(),  # 加载系统提示词（定义AI角色和能力）
            # 工具列表：AI可调用的功能，每个工具都是一个装饰器包装的函数
            tools=[rag_summarize,      # RAG知识库检索工具
                   search_devices,     # 设备检索工具
                   filter_by_budget,   # 预算筛选工具
                   match_by_scene,     # 场景匹配工具
                   sort_by_value],     # 性价比排序工具
        )

    def execute_stream(self, query: str):
        """
        流式执行用户查询
        
        【为什么用流式输出】
        大模型生成回答需要时间，流式输出可以让用户实时看到AI在"打字"，
        提升用户体验，避免长时间等待的焦虑感。
        
        【参数】
        query: 用户的输入问题（如"推荐一款3000元以内的手机"）
        
        【返回值】
        生成器（generator），逐个yield AI生成的内容片段
        
        【生成器是什么】
        生成器是一种特殊的迭代器，用yield代替return，可以暂停执行并保留状态，
        下次从暂停处继续。适合处理流式数据（如大模型输出）。
        """
        # 【为什么包装成input_dict】
        # LangChain的agent需要特定格式的输入，包含messages列表
        # 每个消息是一个字典，包含role（角色）和content（内容）
        input_dict = {
            "messages": [
                {"role": "user", "content": query},  # 用户消息
            ]
        }

        # 【stream方法参数说明】
        # - input_dict: 输入数据，包含用户消息
        # - stream_mode="values": 流式输出模式，返回中间状态值
        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            # 【chunk的结构】
            # chunk是一个字典，包含messages列表，表示当前对话状态的所有消息
            # 包括：系统提示词、用户输入、AI思考过程、工具调用结果、AI最终回答等
            latest_message = chunk["messages"][-1]  # 取最后一条消息（最新的）
            
            # 【为什么过滤type == "ai"】
            # LangChain的stream会输出所有消息类型：user、ai、tool、system等
            # 但我们只关心AI生成的最终回答，不关心中间思考过程或工具结果
            # type="ai"表示这是AI生成的消息，不是用户输入或工具输出
            if latest_message.content and latest_message.type == "ai":
                # 【为什么加strip()和\n】
                # strip()去除首尾空白，\n添加换行，让输出格式更规范
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    # 【为什么写main块】
    # 当直接运行这个文件时（python react_agent.py），执行以下测试代码
    # 当被import时（from agent.react_agent import ReactAgent），不执行
    # 这是Python的标准做法，用于模块自测
    
    agent = ReactAgent()

    # 【测试execute_stream】
    # 调用execute_stream获取生成器，用for循环逐个获取内容片段
    # end=""表示打印不换行，flush=True表示立即刷新缓冲区（实时显示）
    for chunk in agent.execute_stream("推荐一款3000元以内的手机，主要用来打游戏"):
        print(chunk, end="", flush=True)

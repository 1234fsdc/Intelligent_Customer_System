"""
智扫通机器人智能客服系统 - 主应用入口

【为什么用Streamlit】
Streamlit是一个快速构建数据应用和AI界面的Python库，无需前端知识即可创建交互式Web界面。
它自动处理UI组件的渲染和状态管理，非常适合快速原型开发和内部工具。

【整体架构设计】
采用"用户输入→状态更新→rerun→AI回复→rerun"的两阶段处理模式，
解决Streamlit脚本从上到下顺序执行导致的UI更新问题。
"""

import time

import streamlit as st
from agent.react_agent import ReactAgent

# ==================== 页面标题设置 ====================
# 【为什么】Streamlit的st.title会在页面顶部渲染一个标题，让用户知道这是什么应用
# 【怎么做】调用st.title()传入字符串，Streamlit自动渲染为HTML标题元素
st.title("电子设备推荐系统")
st.divider()  # 添加分割线，美化界面

# ==================== Session State 初始化 ====================
# 【为什么】Streamlit是状态less的，每次交互都会重新执行整个脚本。
# 使用st.session_state可以跨脚本执行保持数据（如对话历史、AI实例）。
# 如果不保存，每次点击都会丢失之前的对话。

# 【为什么检查"agent"】避免重复创建ReactAgent实例（创建一次即可复用）
# 【怎么做】用in关键字检查session_state字典中是否存在该键
if "agent" not in st.session_state:
    # 【为什么】ReactAgent初始化需要加载模型、配置等，比较耗时
    # 【怎么做】创建实例并存入session_state，后续直接使用
    st.session_state["agent"] = ReactAgent()

# 【为什么检查"message"】避免每次rerun都清空对话历史
# 【怎么做】初始化为空列表，后续用append添加消息
if "message" not in st.session_state:
    st.session_state["message"] = []

# ==================== 渲染历史消息 ====================
# 【为什么】需要在页面上显示之前的对话记录
# 【怎么做】遍历session_state中的message列表，根据role渲染不同样式的消息
# st.chat_message("user")显示用户头像（红色），st.chat_message("assistant")显示AI头像（黄色）
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# ==================== 用户输入处理 ====================
# 【为什么】st.chat_input()创建一个固定在底部的输入框，用户输入后返回输入内容
# 【怎么做】调用st.chat_input()，当用户输入并回车后，返回输入字符串，否则返回None
prompt = st.chat_input()

# 【为什么】需要判断prompt是否有值（用户是否输入了内容）
if prompt:
    # 【为什么】将用户消息存入session_state，而不是直接显示
    # 【怎么做】用append添加字典，包含role和content两个字段
    # role="user"表示这是用户消息，会在rerun后由上面的循环渲染
    st.session_state["message"].append({"role": "user", "content": prompt})
    
    # 【为什么】st.rerun()强制Streamlit重新执行整个脚本
    # 【怎么做】调用后立即中断当前执行，从头开始执行
    # 【效果】重新执行后，上面的for循环会渲染出刚添加的用户消息
    st.rerun()

# ==================== AI回复处理 ====================
# 【为什么】需要检测是否需要AI回复（即最后一条消息是否是用户发的）
# 【怎么做】检查message列表非空且最后一条的role是"user"
# 这样可以避免AI重复回复或在没有用户输入时自动回复
if st.session_state["message"] and st.session_state["message"][-1]["role"] == "user":
    # 【为什么】需要缓存AI的完整回复，用于后续存入session_state
    response_messages = []
    
    # 【为什么】st.spinner显示加载动画，提升用户体验，让用户知道AI在思考
    # 【怎么做】用with语句包裹AI处理逻辑，进入时显示spinner，结束时自动消失
    with st.spinner("智能客服思考中..."):
        # 【为什么】调用agent的execute_stream获取流式输出
        # 【怎么做】传入用户最后一条消息的内容，返回一个生成器（generator）
        # 生成器可以逐个yield内容片段，实现打字机效果
        res_stream = st.session_state["agent"].execute_stream(st.session_state["message"][-1]["content"])

        # 【为什么】需要包装生成器，在流式输出的同时缓存完整内容
        # 【怎么做】定义一个生成器函数capture，接收原始生成器和缓存列表
        def capture(generator, cache_list):
            # 【为什么】遍历生成器获取每个内容片段
            for chunk in generator:
                # 【为什么】将片段存入缓存列表，用于后续保存完整回复
                cache_list.append(chunk)

                # 【为什么】对每个字符添加延迟，实现打字机效果
                # 【怎么做】遍历chunk中的每个字符，等待0.01秒后yield
                # 这样Streamlit的write_stream会逐个字符显示，模拟打字效果
                for char in chunk:
                    time.sleep(0.01)
                    yield char

        # 【为什么】st.chat_message("assistant")创建AI消息气泡
        # 【怎么做】.write_stream()接收一个生成器，实时显示流式内容
        # 这里传入包装后的capture生成器，既显示流式内容又缓存完整回复
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        
        # 【为什么】将AI的完整回复存入session_state，保存对话历史
        # 【怎么做】取缓存列表的最后一个元素（即完整回复），添加为assistant角色
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        
        # 【为什么】再次rerun，让上面的for循环渲染出AI的回复
        # 【效果】重新执行后，历史消息循环会显示完整的对话记录
        st.rerun()

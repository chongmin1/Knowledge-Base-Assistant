from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

import streamlit as st
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys

def get_retriever():
    """
    :return: 检索器
    """
    embedding = QianfanEmbeddingsEndpoint()
    persist_directory = "./"
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

def combine_docs(docs):
    """
    处理检索器返回的文本
    """
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    """
    :return: 检索问答链
    """
    retriever = get_retriever()
    llm = QianfanChatEndpoint(temprature=0)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "chat_history"),
        ("human", "{input}")
    ])

    # 根据条件选择要运行的分支
    retrieve_docs = RunnableBranch(
        # 聊天记录中没有chat_history则直接使用用户问题查向量库
        (lambda x: not x.get("chat_history", False), (lambda x :x["input"]) | retriever, ),
        # 聊天记录有chat-history就让llm根据聊天记录完善问题再查向量库
        condense_question_prompt | llm | StrOutputParser() | retriever
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("placeholder","{chat_history}"),
        ("human","{input}")
    ])
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    qa_history_chain = RunnablePassthrough.assign(
        context = retrieve_docs
    ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    """
    :param chain: 检索问答链
    :param input: 用户输入
    :param chat_history: 聊天记录
    :return: 流式返回结果
    """
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:  # 遍历对话历史
        with messages.chat_message(message[0]):  # messages指在容器下显示，chat_message显示用户及ai头像
            st.write(message[1])  # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))
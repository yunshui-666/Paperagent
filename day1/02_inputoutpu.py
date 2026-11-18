'''
输入输出隔离
规范什么状态作为输入什么作为输出
通过不同状态定义，隔离，
'''
from typing import TypedDict
from  langgraph.constants import  START,END
from langgraph.graph import StateGraph

class Mystate(TypedDict):
    query:str
    rag_result:str
    web_result:str
    final_result:str

class InputSchema(TypedDict):
    query:str

class Output(TypedDict):
    final_result:str

#构建节点

def rag_search_node(state:Mystate):
    query = state['query']
    rag_result = f"这是结果"
    return {'rag_result': rag_result}

def web_search_node(state:Mystate):
    query = state['query']
    web_result = (f'')

    return {'web_result': web_result}

def final_search_node(state:Mystate):
    query = state['query']
    final_result = (f'')

    return {'final_result': final_result}

#创建一个构造器
builder = StateGraph(state_schema=Mystate),
input_schema = InputSchema()
output_schema = Output()

builder.add_node(rag_search_node)
builder.add_node(final_search_node)
from typing import TypedDict
from unittest import result

from langgraph.constants import START,END
from langgraph.graph import StateGraph

# state声明
class MYstate(TypedDict):
    query: str
    file_result:str
    web_result:str
    final_result:str

def query_web(state:MYstate)->dict:
    query = state['query']
    webresult = f'{query}的搜索结果'
    return {'web_result': webresult}

def query_file(state:MYstate)->dict:
    query = state['query']
    queryresult = f'{query}的文件结果返回'
    return {'file_result': queryresult}

def answer(state:MYstate)->dict:
    web_result = state['web_result']
    file_result = state['file_result']
    final_result = f'基于{web_result}和{file_result}的最终结果'
    return {'final_result': final_result}

graph = StateGraph(state_schema=MYstate)
graph.add_node(answer)
graph.add_node(query_web)
graph.add_node(query_file)
graph.add_edge(START,'query_web')
graph.add_edge(START,'query_file')
graph.add_edge('query_web','answer')
graph.add_edge('query_file','answer')
graph.add_edge('answer',END)

compile_graph = graph.compile()
res = compile_graph.invoke({'query':'什么是langgraph'})

print(res['final_result'])




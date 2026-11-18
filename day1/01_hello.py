# 演示Langgraph
# 定义状态，class，管理数据
"""
流程还是很简单的

"""
from typing import TypedDict


class Mystate(TypedDict):
    query:str
    rag_result:str
    web_result:str
    final_answer:str


# 定义节点，节点就是对数据进行处理的函数或者方法
def rag_search_node(state:Mystate):
    # 获取所需查询内容
    query = state['query']

    rag_result = f'这是基于{query}的处理结果'
    return {"rag_result": rag_result}


def net_search_node(state:Mystate):
    query = state['query']

    net_search_result = f"这是网络搜索的结果{query}"

    return {"net_search_result": net_search_result}


def final_answer(state:Mystate):
    # 从状态中获取数据
    query = state['query']
    rag_result = state['rag_result']
    web_result = state['web_result']

    final_answer = f"这是结果{query}"
    return {"final_answer": final_answer}


#创建构造器对象，构造图结构
builder = StateGraph(state_schema = Mystate)
builder.add_node(rag_search_node)
builder.add_node(net_search_node)
builder.add_node(final_answer)
# 节点连接，通过边连接
builder.add_edge(START,"rag_search_node")
builder.add_edge('rag_search_node', 'net_search_result')
builder.add_edge('net_search_result', 'final_answer')
builder.add_edge('final_answer', END)

graph = builder.compile()
result = graph.invoke({'query':'这是一个测试'})

print(result)

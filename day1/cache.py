import time
from tabnanny import check

from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy
from langgraph.constants import START
from langgraph.checkpoint.memory import InMemorySaver

class MyState(TypedDict):
    x:int
    result:int
builder = StateGraph(MyState)
checkpoint = InMemorySaver()

def expensive_node(state: MyState) -> dict[str,int]:
    print('expensive node被使用')
    time.sleep(1)
    print('expensice_node使用完成')
    return  {'result':state['x']*2}

builder.add_node("expensive_node", expensive_node,
cache_policy = CachePolicy(ttl=10))
builder.add_edge(START,"expensive_node")

graph =  builder.compile(cache=InMemoryCache(),checkpointer = checkpoint)

print(graph.invoke({"x":5},config={"configurable":{"thread_id":"1"}}))

print(graph.invoke({"x": 5},config={"configurable":{"thread_id":"2"}}))
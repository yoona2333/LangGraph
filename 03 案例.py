import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage

load_dotenv()
# 校验是否成功读取密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY 未读取到，请检查 .env 文件！")

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    api_key=api_key,
    extra_body={
        "thinking": {"type": "disabled"}
    }
)

# MessagesState 自带 messages 字段；额外扩展 username、output
class OverAllState(MessagesState):
    username: str
    output: str

def node_a(state: OverAllState)->OverAllState:
    return {
        "messages": [
            HumanMessage("你好，我是" + state["username"])
        ]
    }

def llm_node(state: OverAllState)->OverAllState:
    res = model.invoke(state["messages"])
    return {
        "messages": [res],
        "output": res.content
    }

builder = StateGraph(OverAllState)
builder.add_node("node_a", node_a)
builder.add_node("llm_node", llm_node)

builder.add_edge(START, "node_a")
builder.add_edge("node_a", "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()
result = graph.invoke({"username": "小黄"})
print(result)






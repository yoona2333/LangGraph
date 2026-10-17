# import psycopg
#
# DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
#
# conn = psycopg.connect(DB_URL)
# cur = conn.cursor()
#
# # 查看 PostgreSQL 版本信息
# cur.execute("SELECT version();")
# result = cur.fetchone()
# print(f"PostgreSQL 版本：{result[0]}")
#
# cur.close()
# conn.close()
#

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(MessagesState):
    output: str

def llm_node(state: OverAllState) -> OverAllState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {
        "messages": [res]
    }

def output_node(state: OverAllState) -> OverAllState:
    return {
        "output": state["messages"][-1].content
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("llm_node", llm_node)
builder.add_node("output_node", output_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "output_node")
builder.add_edge("output_node", END)

# 定义并在编译时传递 Checkpointer
#DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
DB_URL = "postgresql://postgres:123456@localhost:5433/langgraph_db?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    # 示例中为了方便演示直接调用 setup()
    # 实际项目中通常建议把数据库初始化/迁移作为独立步骤处理
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    # 定义配置对象
    config = {"configurable": {"thread_id": "chapter_6_6.2.4"}}
    # 调用时传递
    graph.invoke({"messages": [HumanMessage("你好，我是老王")]}, config=config)
    graph.invoke({"messages": [HumanMessage("从现在开始，你是小王")]}, config=config)
    res = graph.invoke({"messages": [HumanMessage("我是谁？你是谁？")]}, config=config)
    print(res["output"])

    print('=' * 30, '-> 完整消息列表 <-', '=' * 30)
    for msg in res["messages"]:
        msg.pretty_print()
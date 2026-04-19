import operator
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,END,START
from typing_extensions import TypedDict

INDEX_DIR = "faiss_index"


load_dotenv()


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llms_calls:int

@tool
def add(a:int,b:int)->int:
    """Adds a to b"""
    return a+b

@tool
def multiply(a:int,b:int)->int:
    """Multiply a to b"""
    return a*b

@tool
def Divide(a:int,b:int)->int:
    """Divide a to b"""
    return a/b

@tool
def search_docs(query:str)->str:
    """Search the knowledge base for information about AML/ML concepts,
    LangGraph, RAG, embeddings, transaformers, and related topics"""
    embeddings= GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore=FAISS.load_local(INDEX_DIR,embeddings,allow_dangerous_deserialization=True)
    docs=vectorstore.as_retriever(search_kwargs={"k":3}).invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


tools= [add, multiply, Divide, search_docs]
tools_by_name = {t.name:t for t in tools}

model = ChatGroq(model="llama-3.1-8b-instant")
model_with_tools = model.bind_tools(tools)

def llm_call(state:MessageState)->dict:
    response = model_with_tools.invoke(
        [SystemMessage(content="""
        You're a helpful assistant that can perform arithmetic and answer questions about AI/ML concepts.
        Use search_docs for AI/ML questions, math tool for calculations.
        """)] + state["messages"]
    )

    return{
        "messages": [response],
        "llms_calls": state.get("llm_calls",0) + 1,
    }

def tool_node(state: MessageState) -> dict:
    results=[]
    for tool_call in state['messages'][-1].tool_calls:
        t=tools_by_name[tool_call['name']]
        observation = t.invoke(tool_call['args'])
        results.append(ToolMessage(content=str(observation),tool_call_id=tool_call['id']))
    return {"messages": results}


def should_continue(state: MessageState)-> Literal["tool_node","__end__"]:
    last = state["messages"][-1]
    if hasattr(last,"tool_calls") and last.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessageState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START,"llm_call")
agent_builder.add_conditional_edges("llm_call",should_continue,["tool_node", END])
agent_builder.add_edge("tool_node","llm_call")

agent = agent_builder.compile()

 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# --------------------------------------------------------------------------
# WHAT'S DIFFERENT ABOUT THIS AGENT vs. agent.py?
# agent.py can only answer from what the AI model already knows.
# This agent can instead look things up in Acme Corp's own VPN guide by
# using a "tool" — the AI decides when it needs to search, calls the
# search tool, reads the results, and then answers using that info.
# This is often called "RAG" (Retrieval-Augmented Generation).
# --------------------------------------------------------------------------

# Re-create the SAME embeddings model used in knowledge_base.py. This is
# needed to turn a user's question into a number-list (embedding) that
# can be compared against the chunks already stored in the vector store.
# It must match the model used when the vector store was built.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load the vector store we built and saved earlier (see knowledge_base.py)
# back from disk, instead of rebuilding it from the PDF every time.
vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)

# A "RETRIEVER" is just the vector store wrapped so it can be asked
# "give me the most relevant chunks for this text." search_kwargs={"k": 4}
# means: return the 4 most relevant chunks for whatever question is asked.
retriver = vectorstore.as_retriever(search_kwargs={"k": 4})

# A "TOOL" is a function the AI model is allowed to call by itself when
# it decides it needs to, rather than us calling it in code directly.
# create_retriever_tool packages our retriever into a tool named
# "search_vpn_knowledge_base", with a description that tells the AI
# model what the tool is for and when to use it (the AI reads this
# description to decide whether calling it makes sense).
kb_tool = create_retriever_tool(
    retriver,
    name="search_vpn_knowledge_base",
    description=("Search Acme Corp's internal VPN user guide for installation steps, troubleshooting, and configuration details."),
)

# bind_tools tells the AI model "here are the tools you're allowed to
# use." From now on, when we call llm_with_locals, the model can either
# reply normally with text, OR reply by asking to call kb_tool — we then
# run that tool and hand the results back to it (see the graph below).
llm_with_locals = llm.bind_tools([kb_tool])

SYSTEM_PROMPT = '''
ou are a helpful IT support assistant for Acme Corp. \
You assist employees with VPN-related issues.
You have access to Acme Corp's internal VPN Knowledge Base through the \
search_vpn_knowledge_base tool - use it to find accurate, relevant answers \
to employee questions about installation, troubleshooting, or configuration.
Always respond clearly and politely.
Do not offer solutions unrelated to VPN.
'''

# This is the "chatbot" step of our flowchart. Same idea as agent.py:
# add the hidden instructions, ask the AI for a reply, and pass that
# reply along. The difference is llm_with_locals may reply with a
# request to use the search tool instead of a plain text answer — the
# graph below (see add_conditional_edges) is what checks for that and
# routes to the "tools" step when it happens.
def chat_bot(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_locals.invoke(messages)
    return {'messages': [response]}


# The model can return `content` as a plain string, or as a list of
# content blocks (e.g. [{'type': 'text', 'text': '...'}]). This pulls
# out just the human-readable text either way, so we never print the
# raw object.
def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content)

# This flowchart has two steps now instead of one, so the agent can
# loop back and forth between "thinking/replying" and "searching":
#   - "chatbot": the AI thinking/replying step (defined above).
#   - "tools": a built-in LangGraph step (ToolNode) that actually runs
#     whichever tool the AI asked for (here, kb_tool) and hands the
#     search results back into the conversation.
#
# add_conditional_edges("chatbot", tools_condition) is the key part:
# after "chatbot" runs, tools_condition looks at the AI's reply and
# decides where to go next —
#   - if the AI asked to use a tool, go to "tools"
#   - otherwise, the AI gave a final answer, so go to END
#
# add_edge("tools", "chatbot") sends the search results back to the
# chatbot step, so the AI can read them and give a proper answer. This
# creates a loop: chatbot -> tools -> chatbot -> ... -> END, letting the
# AI search as many times as it needs before replying.
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chat_bot)
graph_builder.add_node("tools", ToolNode(tools=[kb_tool]))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
agent = graph_builder.compile()


if __name__ == "__main__":
    print("Agent ready. Type 'quit' to exit.\n")
    conversation = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        conversation.append({'role': "user", 'content': user_input})
        result = agent.invoke({"messages": conversation})
        conversation = result["messages"]
        print("Agent:", extract_text(conversation[-1].content), "\n")
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")
response = llm.invoke("Hello, Gemini")
print(llm)
print(response.content)

# --------------------------------------------------------------------------
# WHAT IS A "GRAPH" HERE?
# Think of the agent as a flowchart. A "graph" is just that flowchart:
# a set of steps (called "nodes") connected by arrows (called "edges")
# that tell the program what to do first, second, third, and so on.
# LangGraph is the library that lets us build this flowchart in Python.
# --------------------------------------------------------------------------

# define graph state
# "STATE" = the information the flowchart carries with it as it moves
# from step to step. You can picture it as a clipboard that gets passed
# along the flowchart, and each step can read from it or write on it.
#
# MessagesState is a ready-made clipboard whose only field is
# "messages": the running list of chat messages (who said what),
# so the conversation history is remembered as it flows through the graph.
from langgraph.graph import MessagesState


# create graph nodes
# A "NODE" is one step/box in the flowchart. Each node is just a normal
# Python function: it receives the current state (the clipboard), does
# something (like calling the AI model), and hands back an update to
# the state.
from langchain_core.messages import SystemMessage

SYSTEM_PROMPT = """You are a helpful, knowledgeable AI assistant.

- Answer clearly and concisely; expand only when the question calls for depth.
- If you don't know something or lack enough information, say so instead of guessing.
- When asked to use a tool, use it rather than answering from memory alone.
- If a request is ambiguous, ask a clarifying question before proceeding.
- Keep responses well-structured and easy to skim (short paragraphs, lists when helpful).
"""

# SYSTEM_PROMPT is a set of instructions given to the AI "behind the
# scenes" — the user never sees it, but it tells the AI how to behave
# (be clear, admit when it doesn't know something, etc.), like a
# briefing a person gets before starting a job.

# This function IS the "chatbot" node described above: it's the one
# and only step in our flowchart. It takes the clipboard (state),
# adds the hidden instructions to the top of the conversation, asks
# the AI model for a reply, and puts that reply back onto the
# clipboard for the next step (or the user) to see.
def chat_bot(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {'messages': [response]}



# connect everything into a graph
# Now we actually draw the flowchart:
#   - StateGraph(MessagesState): start a blank flowchart that uses our
#     "clipboard" (chat history) to pass information along.
#   - START and END are special markers meaning "the flowchart begins
#     here" and "the flowchart finishes here" — every flowchart needs
#     a clear starting point and ending point.
#   - add_node(...) places our "chatbot" step onto the flowchart.
#   - add_edge(...) draws the arrows connecting the steps:
#       START -> chatbot   (begin by running the chatbot step)
#       chatbot -> END     (after the chatbot replies, we're done)
#   - compile() finalizes the flowchart into something runnable,
#     stored in "agent". Right now the flowchart is simple (just one
#     step), but this same pattern is how you'd add more steps later
#     (e.g. a step that searches the web, then a step that replies).
from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chat_bot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
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
        print("Agent:", conversation[-1].content, "\n")
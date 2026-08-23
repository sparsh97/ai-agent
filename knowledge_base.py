from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --------------------------------------------------------------------------
# WHAT IS A "KNOWLEDGE BASE" AND WHY DO WE NEED ALL THIS?
# We want our AI agent to answer questions using OUR OWN documents (like
# a VPN user guide), not just what it already knows. The AI can't read a
# whole PDF on every question, so instead we prepare the document ahead
# of time so it can be *searched* quickly. That preparation is:
#   1. Load the document
#   2. Split it into small "chunks"
#   3. Turn each chunk into an "embedding" (a list of numbers)
#   4. Store those in a "vector store" that can be searched by meaning
# --------------------------------------------------------------------------

# load document
# A "DOCUMENT" here is just one page (or section) of text pulled out of
# the PDF, along with a bit of info about where it came from (like which
# page number). PyPDFLoader reads the PDF file and turns it into a list
# of these document objects — one per page.
loader = PyPDFLoader("vpn_user_guide.pdf")
documents = loader.load()

# split document into chunks
# A "CHUNK" is a small piece of text — a few paragraphs' worth — cut out
# of a document. We split because a whole page (or the whole PDF) is too
# big and unfocused to search well; small chunks let us find the exact
# bit of text that answers a specific question.
#   - chunk_size=1000: aim for chunks of about 1000 characters each.
#   - chunk_overlap=200: let neighboring chunks share 200 characters, so
#     we don't accidentally cut a sentence/idea in half between chunks
#     and lose its meaning.
splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitters.split_documents(documents)

# create embeddings and build the FAISS vector store
# An "EMBEDDING" is a way of turning a piece of text into a list of
# numbers (a "vector") that captures its MEANING. Text with similar
# meaning ends up with similar numbers, even if the words used are
# different. This is what lets us later search by "what is this about"
# instead of just matching exact keywords.
#
# GoogleGenerativeAIEmbeddings is the tool that calls Google's AI to
# convert each chunk of text into one of these number-lists.
#
# Which model to use: "models/gemini-embedding-001" is Google's current
# general-purpose text embedding model — a good default for this kind
# of document search/knowledge-base use case.
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# A "VECTOR STORE" is a database built specifically to hold these
# embeddings and quickly find the ones that are most similar to a new
# piece of text (e.g. a user's question, once it's also turned into an
# embedding). FAISS is a fast, local (no server needed) vector store —
# here we build one from all our chunks and their embeddings, so later
# we can ask it "which chunks are most relevant to this question?"
vectorstore = FAISS.from_documents(chunks, embeddings)


# save the vector store to disk
# This writes the vector store to a local folder ("faiss_index") so we
# don't have to redo the load/split/embed steps every time we want to
# search the knowledge base. Other scripts (like the agent) can later
# load this folder back with FAISS.load_local(...) and start searching
# immediately.
vectorstore.save_local("faiss_index")
print(f"Knowledge base created from {len(documents)} pages, {len(chunks)} chunks.")
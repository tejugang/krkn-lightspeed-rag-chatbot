from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# load and chunk contents of the PDF
loader = PyPDFLoader("data/pod_scenarios.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# embed and store in vector database
embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
vector_store = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

# load LLM (chat model)
#llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key= "openAI-api-key" )
# Load the model
llm = Llama(model_path=model_path)
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile the graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# run in a loop
while True:
    q = input("Ask a question (or type 'exit'): ")
    if q.lower() in ["exit", "quit"]:
        break
    result = graph.invoke({"question": q})
    print("\nAnswer:", result["answer"])

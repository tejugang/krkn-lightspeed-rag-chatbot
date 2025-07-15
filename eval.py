from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import Ollama

#delete
import json


# load and chunk contents of thepytohnPDF

loader1 = PyPDFLoader("data/pod_scenarios.pdf")
loader2 = PyPDFLoader("data/Pod-Scenarios-using-Krknctl.pdf")
loader3 = PyPDFLoader("data/Pod-Scenarios-using-Krkn-hub.pdf")
loader4 = PyPDFLoader("data/Pod-Scenarios-using-Krkn.pdf")

docs1 = loader1.load()
docs2 = loader2.load()
docs3 = loader3.load()
docs4 = loader4.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs1 + docs2+ docs3+ docs4)

# embed and store in vector database
embedding_model = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
vector_store = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

# load LLM (chat model)

#lm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key= "openAI-api-key" )


# Load the model (llama 2.7 model)
'''
llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=1,  
)
'''
#granite
'''
model_id = "ibm-granite/granite-3b-code-base-2k"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto" if device == "cuda" else None)
model.to(device)
model.eval()
'''

llm = Ollama(model="llama3.1", base_url="http://127.0.0.1:11434")



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
    return {"answer": response}

# llama and openaI model
    '''
    response = llm.invoke(messages)
    
    return {"answer": response}
    '''
# Compile the graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()



# Your 30 questions
questions = [
    "What is the primary objective of the 'Pod Scenarios' feature within Krkn-hub on a Kubernetes/OpenShift cluster?",
    "How can Cerberus be integrated with Krkn-hub's pod scenarios to monitor the cluster and determine the success or failure of a chaos injection?",
    "What is the standard Podman command structure for initiating a pod disruption scenario with Krkn-hub, including how to provide the kube config and enable host environment variables?",
    "After a Krkn-hub pod scenario has been initiated, what commands can be used with both Podman and Docker to monitor its ongoing logs and to retrieve its final exit code for pass/fail determination?",
    "What is a significant limitation of the `--env-host` option when running Krkn-hub scenarios with Podman on certain client types, and how should environment variables be managed in such cases?",
    "Why is it crucial to adjust the permissions of the kube config file before mounting it into the Krkn-hub container, and what specific commands are recommended to achieve this?",
    "How can environment variables be passed to a Krkn-hub scenario container to customize its behavior, detailing both the host-based and command-line methods?",
    "What is the function of the `NAMESPACE` parameter in Krkn-hub pod scenarios, and what is its default value, including its support for advanced matching?",
    "Explain the interplay between the `POD_LABEL` and `NAME_PATTERN` parameters in determining which pods are targeted for disruption, and what their default behaviors are.",
    "What do the `DISRUPTION_COUNT`, `KILL_TIMEOUT`, and `EXPECTED_RECOVERY_TIME` parameters control in a Krkn-hub pod scenario, and what are their default values?",
    "Beyond specific namespace targeting, how can the `NAMESPACE` environment variable be configured to randomly disrupt pods in OpenShift system namespaces, and what additional mode can be enabled for continuous reliability testing?",
    "When `CAPTURE_METRICS` or `ENABLE_ALERTS` are active, how should custom metrics and alerts profiles be provided to the Krkn-hub container, including the specific internal paths for mounting?",
    "When executing a `pod-scenarios` run with `krknctl`, what are the different mechanisms available for precisely identifying the target pods for disruption, and how do they interact?",
    "Describe the two critical timeout parameters in `krknctl run pod-scenarios` and explain their distinct roles in determining the scenario's outcome.",
    "If a user wants to disrupt more than one pod in a `pod-scenarios` run, which parameter needs to be adjusted, and what is the default behavior if this parameter is not explicitly set?",
    "What is the default scope for targeting pods in a `krknctl run pod-scenarios` execution if no specific namespace, label, or name pattern is provided by the user?",
    "How can a user obtain a comprehensive list of all configurable options and their descriptions for the `krknctl run pod-scenarios` command?",
    "Differentiate between the 'Single Pod Deletion' and 'Multiple Pods Deleted Simultaneously' chaos scenarios regarding their simulated failure types and the primary customer impact each aims to validate or expose.",
    "Explain the significance of 'Pod Eviction' as a chaos scenario, detailing its typical triggers and the specific configurations that are crucial for ensuring zero customer impact during such an event.",
    "What specific metrics and observations from Krkn telemetry are crucial for quantitatively assessing the high availability of an application after a chaos engineering run?",
    "Beyond automatic recovery, what are the key architectural and operational indicators, as described in the text, that confirm an application is highly available in a Kubernetes environment?",
    "How does the 'Single Pod Deletion' scenario specifically validate the resilience mechanisms of Kubernetes, and what is the expected recovery timing for stateless applications in this context?",
    "What is the primary concern regarding customer impact when 'Multiple Pods Deleted Simultaneously' occurs, and what architectural characteristic helps mitigate this impact to ensure high availability?",
    "Describe the role of `topologySpreadConstraints` in achieving high availability and how their effective implementation can be verified using standard Kubernetes commands.",
    "In the context of chaos engineering, what does 'Recovery Is Automatic' signify as a high availability indicator, and why is the absence of manual intervention considered critical for true HA?",
    "What is the primary purpose of the `pod_disruption_scenarios` section within a Krkn configuration, and how is a specific scenario file referenced?",
    "How can a user leverage the provided schema file to enhance their experience when creating a Krkn scenario configuration in an IDE?",
    "Based on the example configuration, which specific Kubernetes component is targeted for disruption, and what criteria are used to select its pods?",
    "What is the function of the `krkn_pod_recovery_time` parameter in a Krkn pod disruption scenario, and what is its value in the provided example?",
    "Beyond a general 'basic pod scenario,' what are some of the critical Kubernetes or OpenShift components for which Krkn offers pre-defined chaos scenarios, and are they currently functional?",
    "If a user wanted to specifically target the Kubernetes API server for a chaos experiment, which Krkn scenario would be appropriate, and what action would it perform?",
    "How does the 'OpenShift System Pods' chaos scenario differ in its targeting approach compared to component-specific scenarios like 'Etcd' or 'Prometheus'?",
    "What is the purpose of the `id` field within a Krkn scenario configuration, as demonstrated by the `kill-pods` example?",
    "Can the `kill-pods` scenario, as configured in the example, disrupt pods located in namespaces other than `kube-system`?",
    "Describe the hierarchical structure for defining chaos scenarios within the top-level `kraken` configuration block.",


]

evaluation_data = []

for q in questions:
    start_time = time.time()
    result = graph.invoke({"question": q})
    end_time = time.time()
    duration = end_time - start_time

    # Extract context
    retrieved_context = "\n\n".join(doc.page_content for doc in result["context"])
    
    evaluation_data.append({
        "user_input": q,
        "generated_answer": result["answer"],
        "retrieved_context": retrieved_context,
        "time_taken_seconds": round(duration, 2)
    })

# Save to file
with open("rag_evaluation_data2.json", "w") as f:
    json.dump(evaluation_data, f, indent=2)

print("Evaluation data saved to 'rag_evaluation_data2.json'")

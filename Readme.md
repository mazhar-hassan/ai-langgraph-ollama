# LangGraph Tutorial

We are using Ollama as our LLM (to run on local machine) and the model we will be using is `llama3.2:1b`

> Read following before proceeding as this is hello-world tutorial for LangGraph
## Important Concepts
### LLM
* LLM is Large Language Model
* It is advanced AI systems that understand and generate human-like text 
* Examples: OpenAI, Google Gemma, Google Gemini, Meta Llama models etc

### Application
It could be any application that can provide interface for user input (or trigger point) which then interact with LLM

ChatGPT is a conversational AI chatbot developed by OpenAI, known for its ability to generate human-like text and engage in conversations

We will write out own application that will provide interface to use LLM

### Ollama
Ollama is a tool that simplifies running and managing large language models (LLMs) locally on your computer

    Ollama itself is not LLM but help in running LLM, you can run multipel LLM using Ollama (Download and install it)

### Models
There are multiple free models (LLMs) that you can download and run with the help of Ollama

`ollama pull llama3.2:1b`

In above example we are using llama3.2 LLM which is trained/support 1 billion parameters you can use its advanced version with more parameters, but obviously it will take more space and some impact on CPU as well.


### Our flow

`Python code` -> `ollama` -> `llama3.2`


## Install Dependencies

### Install Ollama
Download from official site and run it

### Install Models
On Ollama site you can find a list of models, select that best suites you and download them by simple command.

`ollama pull llama3.2:1b`

### Install Python dependencies

`pip install langchain-ollama langgraph pydantic python-dotenv`

### Flow

![LangGraph Flow](flow-lang-graph.png)

import sys, time, torch, numpy

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

class LLMOutHandler(BaseCallbackHandler):
  def __init__(self, device):
    self.tokenstring = ""
    self.device = device

  def on_llm_new_token(self, token, **kwargs) -> None:
    self.tokenstring += token;

def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print("Device used for LLM: %s" % device)
  start = time.time()

  print("Loading Callback Manager")
  LLMout = LLMOutHandler(device)

  print("Loading embeddings")
  embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": device})

  print("Loading LlamaCpp")
  llm = LlamaCpp(model_path = "c:/LLM/llama-2-7b-chat.ggmlv3.q4_0.bin", 
                 n_ctx = 2048,
                 n_gpu_layers = 100,
                 n_batch = 512,
                 n_threads = 1,
                 top_k = 10000,
                 temperature = 0.7,
                 max_tokens = 2000,
                 callback_manager = CallbackManager([LLMout]), verbose = False)
  print("Setup took %d seconds" % round(time.time() - start, 2))

  template = """Summarize the following text: '{text}', respond as Snoop Dogg"""
  prompt = PromptTemplate(template = template, input_variables = ["text"])
  chain = LLMChain(llm = llm, prompt = prompt, verbose = False)

  # Input some text
  text = input("Enter a query: ")

  # Run the chain
  chain.run(text = text)
  print("Response after %d seconds" % round(time.time() - start, 2))
  print(LLMout.tokenstring, end = '')

if __name__ == "__main__":
  main()
  
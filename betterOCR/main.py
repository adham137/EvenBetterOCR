from llm.groq_llm import GroqLLM

llm = GroqLLM()
prompt = 'hi'
print(llm.run(prompt))
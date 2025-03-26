# Whole-system-Masters-Project


# Frame work for fine tuned LLM Evaluation 
Cosine similarity 
Ask chat gpt, gemini and claude which is the best on a set on criteria

Relevance – Does the report address the research question?
Coherence – Is the argument well-structured and logically sound?
Accuracy – Are the facts correct and well-supported?
Completeness – Does it cover all key aspects of the topic?
Readability – Is the language clear and concise?
Novelty – Does the report provide new insights or perspectives?

# Required to run this script 
 transformers torch autogen json5 numpy textstat sentence-transformers nltk huggingface_hub python-dotenv datasets peft faiss-cpu bitesandbytes

# Steps

step 1 = 20 research questions created, report written for each question using papers extracted from faiss database
step 2 = Agent systems created feedback for each section and saves to a json 
step 3 = Report is rewritten with the feedback
step 4 = Smaller model is fine tuned with the reports


# Whole-system-Masters-Project


# Steps

* step 1 = 20 research questions created, report written for each question using papers extracted from faiss database
* step 2 = Agent systems created feedback for each section and saves to a json 
* step 3 = Report is rewritten with the feedback
* step 4 = Smaller model is fine tuned with the reports
* Evaluation framework 
  

# Report Evaluation

The metrics calculated are: 
* Word Count     
* Flesch Reading Ease Score  
* Technical Term Count  
* Defined Terms Count  
* Example count  
* Coherence Metrics:  
* Contextual Coherence: Evaluates how well ideas flow and connect throughout the text using sentence embeddings to measure semantic relationships between paragraphs.  
* Measures the smooth progression of ideas by analyzing local coherence between adjacent paragraphs and overall thematic consistency.  
* LLM as a judge for technical depth and clarity
* Measures how clearly topics are sepearted and how indepth sentences are 
Weighted Composite Scores:  

The final evaluation combines these metrics into three main categories with specific weightings:
Technical Depth (45%): Combines technical term count, concept hierarchy depth, and LLM technical depth assessment.  
Clarity & Understandability (35%): Incorporates Flesch score, defined terms, examples, and LLM clarity assessment.  
Structure (20%): Includes coherence flow score  

![image](https://github.com/user-attachments/assets/b1ad2aa7-c1cc-4a69-829b-4ccde9c74a92)


![image](https://github.com/user-attachments/assets/ff800367-c48c-41e6-9cbf-692e40ae1ecf)


# LLM evaluation

* Golden standard answer from  
* Cosine similarity and movers distance
* LLM as a judge 

![image](https://github.com/user-attachments/assets/b923b311-2f07-42c1-80f9-376ac4fcf216)





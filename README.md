# Whole-system-Masters-Project


# Steps

step 1 = 20 research questions created, report written for each question using papers extracted from faiss database
step 2 = Agent systems created feedback for each section and saves to a json 
step 3 = Report is rewritten with the feedback
step 4 = Smaller model is fine tuned with the reports

# Report Evaluation

The metrics calculated are: 
Word Count: A simple measure of document length, comparing the number of words between original and improved versions.
Flesch Reading Ease Score: A standard readability metric that evaluates text complexity based on sentence length and syllable count. Scores range from 0-100, where higher scores indicate easier readability.
Technical Term Count: Tracks the frequency of domain-specific terminology related to semiconductors and technical writing, using a predefined list of technical terms.
Concept Hierarchy Depth: Analyzes the hierarchical structure of ideas in the text using a combination of topic modeling (LDA) and syntactic analysis. Scores range from 1-5, where higher scores indicate more complex concept organization.
Defined Terms Count: Identifies instances where technical terms are explicitly defined or explained in the text.
Example Count: Measures the number of examples provided in the text by identifying specific linguistic patterns that introduce examples.
Actionable Recommendations Count: Identifies concrete suggestions or recommendations in the text using pattern matching for directive language.
Coherence Metrics:
Contextual Coherence: Evaluates how well ideas flow and connect throughout the text using sentence embeddings to measure semantic relationships between paragraphs.
Flow Score: Measures the smooth progression of ideas by analyzing local coherence between adjacent paragraphs and overall thematic consistency.
LLM (Claude) Evaluation Metrics:
Technical Depth Score: An AI-generated score (0-100) assessing the technical sophistication of the content.
Clarity Score: An AI-generated score (0-100) evaluating how well the content is explained and structured.
Overall Score: An AI-generated comprehensive score (0-100) combining various aspects of the document's quality.
Weighted Composite Scores:
The final evaluation combines these metrics into three main categories with specific weightings:
Technical Depth (45%): Combines technical term count, concept hierarchy depth, and LLM technical depth assessment.
Clarity & Understandability (35%): Incorporates Flesch score, defined terms, examples, and LLM clarity assessment.
Structure (20%): Includes coherence flow score and actionable recommendations count.

![image](https://github.com/user-attachments/assets/b1ad2aa7-c1cc-4a69-829b-4ccde9c74a92)


![image](https://github.com/user-attachments/assets/ff800367-c48c-41e6-9cbf-692e40ae1ecf)


# LLM evaluation





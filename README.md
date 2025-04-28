## Generation of Electronic Engineering Research Books Using LLMs and Multi Agent Refinement

This repo is an application that generates research books with 20 chapters using a database of markdown papers.

## Set Up

A conda environment must be set up with the evironment.yml file.

In an .env file a hugging face and API token must be set up as  HUGGINGFACE_TOKEN=INSERT_TOKEN and OPENAI_API_KEY=INSERT_TOKEN

The code is set up to utilise GPUs as shown in the slurm job. This script shows the order to run the scripts in

## Scripts

# Step 1
Creates a faiss database from the markdowns. It is stored under embeddings folder after creation.

# Step 2
Generates chapters using RAG and the created the database and saves them under initial chapters. A repition check takes place and if created chapters are too similar to previous they are rewritten.

# Step 3
This process involves taking the initial chapters and putting them through the agent reviewing process. After the iterations the final chapters are saved in a markdown file called chapters_markdown

# Step 4
This step creates relevant markdown files for each chapter and saves them in chapter_diagrams to be rendered outside the system


The overall system can be seen below:

[finalfinal (41).pdf](https://github.com/user-attachments/files/19932539/finalfinal.41.pdf)







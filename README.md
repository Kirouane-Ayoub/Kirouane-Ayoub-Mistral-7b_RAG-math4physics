# Mistral-7b_RAG-math4physics Chatbot

## Description:

**Mistral-7b_RAG-math4physics** is an advanced chatbot designed to provide accurate and informative responses to questions in the fields of **mathematics and physics**. It leverages state-of-the-art technology, including the **Mistral-7B** language model, **RAG methodology, Chroma DB** for vector storage, and a user-friendly **Streamlit GUI**. With access to the arxiv-math dataset by **TuningAI TEAM**, this chatbot is a powerful tool for students, educators, and enthusiasts seeking answers to complex math and physics inquiries.



## Key Components:

+ **Mistral-7B Language Model**: A **state-of-the-art 7.3 billion** parameter language model developed by **Mistral AI**. It excels in generating human-like text and understanding context.
+ **RAG Method**: The Retrieval-Augmented Generation approach combines the strengths of a retrieval model and a generative language model to provide accurate and factually sound responses.
+ **Chroma DB**: An open-source vector store database optimized for efficient storage and retrieval of vector embeddings, which enhances the chatbot's ability to find relevant information.
+ **arxiv-math Dataset**: Accesses a comprehensive dataset of more than **50,000 math and physics questions and answers** curated by **TuningAI TEAM** for robust knowledge.
+ **Embedding Model**: The **all-MiniLM-L6-v2** model serves as the embedding model, facilitating the extraction of meaningful representations from text data.
+ **Streamlit GUI**: A user-friendly interface powered by Streamlit, making it easy for users to interact with the chatbot and ask mathematical questions.

## Use Cases:

* Homework and Assignment Assistance
* Concept Clarification
* Exam Preparation
* Research Support
* Math and Physics Enthusiast Queries


## Technologies Used:

+ **Mistral-7B** Large Language Model
+ **Retrieval-Augmented Generation (RAG)** Methodology
+ **Chroma DB** for Vector Embedding Storage
+ **Streamlit** for Graphical User Interface **(GUI)**
+ **arxiv-math** Dataset by **TuningAI TEAM**

## Features:

+ Math and Physics Enthusiast Queries
+ Fact-based responses thanks to RAG's retrieval capabilities.
+ Extensive dataset for robust training.
+ Efficient vector embedding storage and retrieval.
+ User-friendly interface for easy interaction.


## Usage : 

```
pip install -r requirements.txt
streamlit run app.py
```

![Screenshot at 2023-10-06 13-39-58](https://github.com/Kirouane-Ayoub/Mistral-7b_RAG-math4physics/assets/99510125/2f62c710-e0e6-4722-9467-2685edd6b574)


## Troubleshooting : 

+ If you see the following error:
```
Traceback (most recent call last):
File "", line 1, in
File "/transformers/models/auto/auto_factory.py", line 482, in from_pretrained
config, kwargs = AutoConfig.from_pretrained(
File "/transformers/models/auto/configuration_auto.py", line 1022, in from_pretrained
config_class = CONFIG_MAPPING[config_dict["model_type"]]
File "/transformers/models/auto/configuration_auto.py", line 723, in getitem
raise KeyError(key)
KeyError: 'mistral'
```

**Installing transformers from source should solve the issue:**

```
pip install git+https://github.com/huggingface/transformers
```




+ **Developed By Kirouane Ayoub**

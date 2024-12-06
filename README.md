# Mini ChatGPT
### _Simple Model, Bidirectional LLM_
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/your-notebook.ipynb)

Here we are going to use the Bidirectional Large Language Model which splits the imported Pdf into text file and run the question answering model with the help of  [BERT] models. This model is very first bidirectional model used by google.

Table of Contents :
1. Installation and Import Libraries 
2. Extract Text from PDFs
3. Preprocess the Text
4. Chunk Large Text (if needed)
5. Question-Answering Model
6. Save The Text 
7. Function to Work with the GPU-Enabled Model
8. Apply the Model 
9. Extra : **Transformers on single paragraph**

## 1. Installation and Import Libraries
```sh
#! pip install PyPDF2 transformers
#! pip install os
#! pip install re
import os
import re
import PyPDF2
from transformers import pipeline
```
> PyPDF2  : 
> transformaers :
> re : 
> pipeline : 

## 2. Extract Text from PDFs (all pages )
```sh
def extract_pdf_text_all_pages(path_of_pdf):
    with open(path_of_pdf, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page_num in range(len(reader.pages)):
		    # for all pages
            page = reader.pages[page_num]
            full_text += page.extract_text() + "\n"
        return full_text
```

# 3. Preprocess the extracted text
```sh
def preprocess_text(text):
    # Remove special characters, extra spaces, or newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
```
>  **Replace multiple spaces with one**
>  **Remove special characters**
# 4. Chunk text into smaller parts to handle large PDFs
```sh
def chunk_text(text, max_length=1000):
    words = text.split()
    chunked_words = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunked_words
```
# 5. Question-Answering Model

[![BiLSTM Image ](https://github.com/amarghimire/BiLSTM-Pdfchat/blob/main/BiLSTM-Figure.png)](https://github.com/amarghimire/BiLSTM-Pdfchat/blob/main/BiLSTM-Figure.png)




We can see one more additional layer than that of the standard LSTM . The additional layer extract the result by analyzing the both sequential flow and give the result which is more convenient .

```sh
from transformers import pipeline
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
qa_model = pipeline("question-answering", model=model_name, device=0)  # device=0 for GPU, device=-1 for CPU

def get_answers_from_text(question, context):
    return qa_model(question=question, context=context)

def process_text_in_chunks(question, text_chunks):
    answers = []
    for chunk in text_chunks:
        result = get_answers_from_text(question, chunk)
        answers.append(result['answer'])
    return answers
```
Load the pre-trained question-answering model (BERT) with GPU support (if available) . Create a function to get answers for a given question and text context. Process the text in chunks if necessary

# 6. Save the answers to a text file
```sh
# Save the answers to a text file
def save_answers_to_file(answers, output_file="answers.txt"):
    with open(output_file, "w") as file:
        for i, answer in enumerate(answers):
            file.write(f"Answer from chunk {i+1}: {answer}\n")
```

# 7. Function to Work with the GPU-Enabled Model
This is modification of the 
```sh
def main(directory_path, question):
    # Step 1: Extract text from all PDFs in the directory
    all_pdfs_text = extract_text_from_multiple_pdfs(directory_path)
    
    # Step 2: Preprocess the text
    preprocessed_text = preprocess_text(all_pdfs_text)
    
    # Step 3: Chunk the text if necessary (if it's too long)
    text_chunks = chunk_text(preprocessed_text, max_length=1000)
    
    # Step 4: Use the question-answering model to get answers from the chunks
    answers = process_text_in_chunks(question, text_chunks)
    save_answers_to_file(answers)
    # Print the answers
    for i, answer in enumerate(answers):
        print(f"Answer from chunk {i+1}: {answer}")
```
# 8. Example usage
```sh
# Example usage
directory_path = './'  # Replace with your PDF folder path
question = "What is Computer Science ? " #you can cange this question according to your pdf. 
main(directory_path, question)
```

# 9. Transformers on single paragraph

```sh
from  transformers  import  pipeline
qa_pipeline = pipeline("question-answering")
paragraph = """For the geometrical optimization of the reactant and product states, and the TS, you should use the B3LYP functional along with the D3 version of Grimme’s dispersion correction with Becke- Johnson damping. You should use the def2-SVP basis set that is of double zeta quality. You should use the SMD solvent model to emulate the experimental conditions. You should employ tight convergence criteria for geometrical optimization and TS search. You can use either the NEB-TS method in ORCA or the QST2+IRC method in Gaussian 16."""

# Example questions

questions = ["What is ORCA",
"What details can you extract from the text?",
"How does the paragraph relate to the topic?",]
for  question  in  questions:
	result = qa_pipeline(question=question, context=paragraph)
	print(f"Question: {question}\nAnswer: {result['answer']}\n")
```
[//]:#
[BERT]: <https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad>

[BILSTM]:<https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm#:~:text=Bidirectional%20LSTM%20(BiLSTM)%20is%20a,utilizing%20information%20from%20both%20sides.>

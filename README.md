# ArIES_Pdf_QnA_app
In the rapidly evolving field of artificial intelligence and natural language processing, the ability to extract meaningful information from documents is becoming increasingly crucial. The "PDF Answering AI" project aims to address this need by leveraging the capabilities of pre-trained and fine-tuned transformer models to provide accurate and contextually relevant answers to user queries based on the content of PDF documents.
# Overview
The project utilizes BERT for Question Answering Model (a bert large uncased model fine-tuned on SQuAD for question answering purposes) as well as DPR models that are used to encode the pdf text as well as the questions to find the passage from the text that is most relevant to the answer of the question asked. The model takes a PDF document and a question from the user as an input and answers the same, from the pdf provided, in real-time.
# Key Features
* ### PDF text extraction
* ### Dense Passage Retrieval (DPR)
* ### BERT for Question Answering

 # How to run on your computer ?
 ## Installation 
 #### 1 Clone this repository:
    git clone https://github.com/protonyo/ArIES_Pdf_QnA_app.git
 #### 2 Navigate to the project directory:
    cd ./ArIES_Pdf_QnA_app/
    next line:
    cd ./flask_qa_app/
 #### 3 Install the required packages:
    pip install -r requirements.txt
    
 ## Run on local Host:
    python app.py

 #### After that a link will be displayed on your terminal , open it and then pdf-answering system will be up on your browser.

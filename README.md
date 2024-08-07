# Building a Full-Stack Chatbot with LangChain and Custom RAG Model

This tutorial guides you through creating a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application includes a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base.

## Table of Contents

- [Building a Full-Stack Chatbot with LangChain and Custom RAG Model](#building-a-full-stack-chatbot-with-langchain-and-custom-rag-model)
  - [Table of Contents](#table-of-contents)
  - [Chapter 1: Introduction](#chapter-1-introduction)
    - [Project Structure](#project-structure)
      - [Step 1: Backend Setup](#step-1-backend-setup)
      - [Step 2: Frontend Setup](#step-2-frontend-setup)
      - [Step 3: Integrate Custom Material for RAG](#step-3-integrate-custom-material-for-rag)
      - [Step 4: Run and Test the Application](#step-4-run-and-test-the-application)
    - [Conclusion for Chapter 1](#conclusion-for-chapter-1)
  - [Chapter 2: Scaling consideration: Scalable Knowledge Base](#chapter-2-scaling-consideration-scalable-knowledge-base)
    - [1. Use a Database for Storage](#1-use-a-database-for-storage)
      - [NoSQL Databases](#nosql-databases)
      - [SQL Databases](#sql-databases)
    - [2. Use Vector Databases for Embeddings](#2-use-vector-databases-for-embeddings)
    - [3. Implement Incremental Learning and Update Mechanisms](#3-implement-incremental-learning-and-update-mechanisms)
    - [4. Indexing and Search Optimization](#4-indexing-and-search-optimization)
    - [Example: Using MongoDB and Faiss for a Scalable Knowledge Base](#example-using-mongodb-and-faiss-for-a-scalable-knowledge-base)
    - [Conclusion for Chapter 2](#conclusion-for-chapter-2)

## Chapter 1: Introduction

This tutorial will guide you through creating a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application will have a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base.

### Project Structure

Create a project directory with the following structure:

```text
rag-chatbot-app/
├── backend/
│   ├── app.py
│   └── rag_model.py
├── frontend/
├── model/
├── data/
│   └── knowledge_base.json
└── README.md
```

#### Step 1: Backend Setup

**Technology Stack**: Python, Flask, MongoDB

1. **Initialize the Backend**

   ```bash
   mkdir backend
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install flask flask-restful pymongo flask-cors langchain transformers torch faiss-cpu
   ```

2. **Create the Flask App**

   ```python
   # backend/app.py
   from flask import Flask, request
   from flask_restful import Resource, Api
   from flask_cors import CORS
   from pymongo import MongoClient
   from langchain import LLMChain
   from langchain.chains import SimpleSequentialChain
   from langchain.prompts import ChatPromptTemplate
   from rag_model import generate_response

   app = Flask(__name__)
   CORS(app)
   api = Api(app)

   # Connect to MongoDB
   client = MongoClient("mongodb://localhost:27017/")
   db = client["rag_chatbot"]

   # Initialize LangChain
   prompt_template = ChatPromptTemplate.from_template("The user said: {user_message}\nThe bot should respond with:")
   chain = SimpleSequentialChain([LLMChain(prompt_template=prompt_template)])

   class Message(Resource):
       def get(self):
           messages = list(db.messages.find())
           return {'messages': messages}, 200

       def post(self):
           user_message = request.json['message']
           db.messages.insert_one({'message': user_message})

           bot_response = chain.run(user_message)
           db.messages.insert_one({'message': bot_response})

           return {'message': bot_response}, 201

   api.add_resource(Message, '/message')

   if __name__ == '__main__':
       app.run(debug=True)
   ```

3. **Create RAG Model Script**

   ```python
   # backend/rag_model.py
   from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
   import torch
   import json
   import faiss
   import numpy as np

   # Load the custom knowledge base
   with open('data/knowledge_base.json') as f:
       knowledge_data = json.load(f)

   # Initialize the tokenizer and model
   tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
   retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom")
   model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

   # Create embeddings for the custom knowledge base
   embeddings = []
   for doc in knowledge_data:
       inputs = tokenizer(doc['text'], return_tensors="pt")
       embeddings.append(model.retrieval_embeddings(**inputs).detach().numpy())

   embeddings = np.vstack(embeddings)
   index = faiss.IndexFlatL2(embeddings.shape[1])
   index.add(embeddings)
   retriever.retrieval_index = index

   def generate_response(query):
       input_ids = tokenizer(query, return_tensors="pt").input_ids
       question_hidden_states = model.question_encoder(input_ids)[0]
       docs_dict = retriever(input_ids, question_hidden_states.numpy(), return_tensors="pt")
       doc_ids = docs_dict["doc_ids"]
       context_input_ids = docs_dict["context_input_ids"]
       outputs = model.generate(input_ids=input_ids, context_input_ids=context_input_ids)
       response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
       return response
   ```

4. **Run the Flask App**

   ```bash
   cd backend
   python app.py
   ```

#### Step 2: Frontend Setup

**Technology Stack**: React

1. **Initialize the Frontend**

   ```bash
   npx create-react-app frontend
   cd frontend
   npm install axios
   ```

2. **Create a Chat Component**

   ```jsx
   // frontend/src/App.js
   import React, { useState, useEffect } from 'react';
   import axios from 'axios';

   const App = () => {
     const [messages, setMessages] = useState([]);
     const [input, setInput] = useState('');

     useEffect(() => {
       axios.get('http://localhost:5000/message')
         .then(response => {
           setMessages(response.data.messages);
         });
     }, []);

     const sendMessage = () => {
       axios.post('http://localhost:5000/message', { message: input })
         .then(response => {
           setMessages([...messages, { message: input }]);
           setInput('');
           setTimeout(() => {
             axios.get('http://localhost:5000/message')
               .then(response => {
                 setMessages(response.data.messages);
               });
           }, 1000); // Allow some time for the bot to respond
         });
     };

     return (
       <div>
         <h1>Chatbot</h1>
         <div>
           {messages.map((msg, index) => <p key={index}>{msg.message}</p>)}
         </div>
         <input
           value={input}
           onChange={(e) => setInput(e.target.value)}
           type="text"
           placeholder="Type a message"
         />
         <button onClick={sendMessage}>Send</button>
       </div>
     );
   };

   export default App;
   ```

3. **Run the React App**

   ```bash
   cd frontend
   npm start
   ```

#### Step 3: Integrate Custom Material for RAG

1. **Prepare Your Knowledge Base**

   Create a `knowledge_base.json` file in the `data/` directory with your custom material:

   ```json
   [
     {"title": "Document 1", "text": "This is the content of document 1."},
     {"title": "Document 2", "text": "This is the content of document 2."}
   ]
   ```

2. **Update RAG Model Script**

   Make sure your `rag_model.py` script correctly references the `knowledge_base.json` file.

#### Step 4: Run and Test the Application

1. **Run the Backend**

   ```bash
   cd backend
   python app.py
   ```

2. **Run the Frontend**

   ```bash
   cd frontend
   npm start
   ```

3. **Test the Chatbot**

   Open your browser and navigate to `http://localhost:3000`. Interact with your chatbot and verify that it uses your custom knowledge base to generate responses.

### Conclusion for Chapter 1

By following this tutorial, you have created a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application includes a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base. LangChain helps manage continuous dialog effectively, providing a seamless conversational experience

## Chapter 2: Scaling consideration: Scalable Knowledge Base

To improve and scale your knowledge base, you should consider using a more robust and scalable solution that can handle large volumes of data efficiently. Here are some techniques and industry-standard approaches for scaling your knowledge base:

### 1. Use a Database for Storage

**Databases** are more scalable and manageable compared to JSON files. Depending on your needs, you can choose between SQL and NoSQL databases.

#### NoSQL Databases

- **MongoDB**: Ideal for storing unstructured data, allows flexible schema design, and supports horizontal scaling.
- **Elasticsearch**: Great for full-text search capabilities and handling large volumes of data with real-time search functionalities.

#### SQL Databases

- **PostgreSQL**: A powerful, open-source relational database that supports advanced indexing, full-text search, and JSON data types.
- **MySQL**: Another popular relational database known for its reliability and ease of use.

### 2. Use Vector Databases for Embeddings

For handling the vector embeddings and efficient similarity searches, consider using vector databases:

- **Faiss** (by Facebook): Highly efficient for similarity search and clustering of dense vectors.
- **Pinecone**: Managed vector database service, designed for production-scale vector similarity search.
- **Weaviate**: An open-source vector search engine with powerful semantic search capabilities.

### 3. Implement Incremental Learning and Update Mechanisms

Ensure your knowledge base can be updated incrementally without needing to rebuild everything from scratch. This involves:

- **Batch Processing**: Regularly update the database with new data in batches.
- **Streaming Updates**: Real-time data ingestion and processing to keep the knowledge base current.

### 4. Indexing and Search Optimization

For efficient retrieval, use indexing and search optimization techniques:

- **Inverted Indexes**: Commonly used in search engines to map content to its locations in the database.
- **TF-IDF**: Term Frequency-Inverse Document Frequency to score the relevance of documents.
- **BM25**: An advanced ranking function used by search engines to rank documents.

### Example: Using MongoDB and Faiss for a Scalable Knowledge Base

1. **Set Up MongoDB**

   Install MongoDB and set up a collection for storing documents.

   ```python
   # backend/app.py (extended)
   from flask import Flask, request
   from flask_restful import Resource, Api
   from flask_cors import CORS
   from pymongo import MongoClient
   from langchain import LLMChain
   from langchain.chains import SimpleSequentialChain
   from langchain.prompts import ChatPromptTemplate
   from rag_model import generate_response, add_document_to_db, update_embeddings

   app = Flask(__name__)
   CORS(app)
   api = Api(app)

   # Connect to MongoDB
   client = MongoClient("mongodb://localhost:27017/")
   db = client["rag_chatbot"]

   # Initialize LangChain
   prompt_template = ChatPromptTemplate.from_template("The user said: {user_message}\nThe bot should respond with:")
   chain = SimpleSequentialChain([LLMChain(prompt_template=prompt_template)])

   class Message(Resource):
       def get(self):
           messages = list(db.messages.find())
           return {'messages': messages}, 200

       def post(self):
           user_message = request.json['message']
           db.messages.insert_one({'message': user_message})

           bot_response = chain.run(user_message)
           db.messages.insert_one({'message': bot_response})

           return {'message': bot_response}, 201

   class Document(Resource):
       def post(self):
           document = request.json['document']
           add_document_to_db(document)
           update_embeddings()
           return {'message': 'Document added and embeddings updated'}, 201

   api.add_resource(Message, '/message')
   api.add_resource(Document, '/document')

   if __name__ == '__main__':
       app.run(debug=True)
   ```

2. **Add and Update Documents in MongoDB**

   ```python
   # backend/rag_model.py (extended)
   from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
   import torch
   import faiss
   import numpy as np
   from pymongo import MongoClient

   # Connect to MongoDB
   client = MongoClient("mongodb://localhost:27017/")
   db = client["rag_chatbot"]
   documents_collection = db['documents']

   # Initialize the tokenizer and model
   tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
   retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom")
   model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

   # Function to add document to the database
   def add_document_to_db(document):
       documents_collection.insert_one(document)

   # Function to update embeddings
   def update_embeddings():
       documents = list(documents_collection.find())
       embeddings = []
       for doc in documents:
           inputs = tokenizer(doc['text'], return_tensors="pt")
           with torch.no_grad():
               embeddings.append(model.retrieval_embeddings(**inputs).detach().numpy())

       embeddings = np.vstack(embeddings)
       index = faiss.IndexFlatL2(embeddings.shape[1])
       index.add(embeddings)
       retriever.retrieval_index = index

   # Function to generate response
   def generate_response(query):
       input_ids = tokenizer(query, return_tensors="pt").input_ids
       question_hidden_states = model.question_encoder(input_ids)[0]
       docs_dict = retriever(input_ids, question_hidden_states.numpy(), return_tensors="pt")
       doc_ids = docs_dict["doc_ids"]
       context_input_ids = docs_dict["context_input_ids"]
       outputs = model.generate(input_ids=input_ids, context_input_ids=context_input_ids)
       response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
       return response
   ```

3. **Client Request to Add Documents**

   ```python
   import requests

   document = {
       "title": "New Document",
       "text": "This is the content of the new document."
   }

   response = requests.post('http://localhost:5000/document', json={'document': document})
   print(response.json())
   ```

### Conclusion for Chapter 2

By following these steps, you can create a scalable knowledge base using MongoDB and Faiss. This approach allows you to handle a large volume of documents efficiently, provides robust search capabilities, and ensures that your chatbot can generate relevant responses using the RAG model

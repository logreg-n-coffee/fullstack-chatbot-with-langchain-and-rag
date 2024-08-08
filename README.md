# Building a Full-Stack Chatbot with LangChain and Custom RAG Model

This tutorial guides you through creating a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application includes a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base.

## Table of Contents

- [Building a Full-Stack Chatbot with LangChain and Custom RAG Model](#building-a-full-stack-chatbot-with-langchain-and-custom-rag-model)
  - [Table of Contents](#table-of-contents)
  - [Discover 1: Introduction](#discover-1-introduction)
    - [Project Structure](#project-structure)
      - [Step 1: Backend Setup](#step-1-backend-setup)
      - [Step 2: Frontend Setup](#step-2-frontend-setup)
      - [Step 3: Integrate Custom Material for RAG](#step-3-integrate-custom-material-for-rag)
      - [Step 4: Run and Test the Application](#step-4-run-and-test-the-application)
    - [Conclusion for Discovery 1](#conclusion-for-discovery-1)
  - [Discover 2: Scaling consideration: Scalable Knowledge Base](#discover-2-scaling-consideration-scalable-knowledge-base)
    - [1. Use a Database for Storage](#1-use-a-database-for-storage)
      - [NoSQL Databases](#nosql-databases)
      - [SQL Databases](#sql-databases)
    - [2. Use Vector Databases for Embeddings](#2-use-vector-databases-for-embeddings)
    - [3. Implement Incremental Learning and Update Mechanisms](#3-implement-incremental-learning-and-update-mechanisms)
    - [4. Indexing and Search Optimization](#4-indexing-and-search-optimization)
    - [Example: Using MongoDB and Faiss for a Scalable Knowledge Base](#example-using-mongodb-and-faiss-for-a-scalable-knowledge-base)
    - [Conclusion for Discovery 2](#conclusion-for-discovery-2)
  - [Discovery 3: Use AWS to Build the Chatbot](#discovery-3-use-aws-to-build-the-chatbot)
    - [Step-by-Step Guide: Using `pgvector` on AWS](#step-by-step-guide-using-pgvector-on-aws)
    - [1. Setting Up the Backend on Amazon EC2](#1-setting-up-the-backend-on-amazon-ec2)
    - [2. Setting Up PostgreSQL with `pgvector` on Amazon RDS](#2-setting-up-postgresql-with-pgvector-on-amazon-rds)
    - [3. Integrate the Backend with RDS and `pgvector`](#3-integrate-the-backend-with-rds-and-pgvector)
    - [4. Setting Up Caching with Amazon ElastiCache](#4-setting-up-caching-with-amazon-elasticache)
    - [5. Storing Static Assets with Amazon S3](#5-storing-static-assets-with-amazon-s3)
    - [6. Using AWS Lambda for Serverless Functions](#6-using-aws-lambda-for-serverless-functions)
    - [7. Setting Up API Gateway](#7-setting-up-api-gateway)
    - [Example: Full Application Code](#example-full-application-code)
      - [backend/app.py](#backendapppy)
      - [backend/rag\_model.py](#backendrag_modelpy)
      - [frontend/src/App.js](#frontendsrcappjs)
    - [Conclusion for Discovery 3](#conclusion-for-discovery-3)

## Discover 1: Introduction

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

### Conclusion for Discovery 1

By following this tutorial, you have created a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application includes a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base. LangChain helps manage continuous dialog effectively, providing a seamless conversational experience

## Discover 2: Scaling consideration: Scalable Knowledge Base

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

### Conclusion for Discovery 2

By following these steps, you can create a scalable knowledge base using MongoDB and Faiss. This approach allows you to handle a large volume of documents efficiently, provides robust search capabilities, and ensures that your chatbot can generate relevant responses using the RAG model

## Discovery 3: Use AWS to Build the Chatbot

Absolutely! Let's integrate `pgvector` within a PostgreSQL instance on AWS RDS while maintaining best practices for deploying a scalable chatbot application on AWS. Here’s a detailed step-by-step guide:

### Step-by-Step Guide: Using `pgvector` on AWS

### 1. Setting Up the Backend on Amazon EC2

1. **Launch an EC2 Instance**

   - Go to the EC2 Dashboard.
   - Click "Launch Instance".
   - Choose an Amazon Machine Image (AMI), e.g., Amazon Linux 2 AMI.
   - Choose an instance type (e.g., t2.micro for testing).
   - Configure instance details (default settings are fine for now).
   - Add storage (default settings are fine).
   - Add tags (optional).
   - Configure security group (allow HTTP, HTTPS, and SSH traffic).
   - Review and launch the instance.

2. **Connect to the EC2 Instance**

   - Use SSH to connect to your instance:

     ```bash
     ssh -i "your-key-pair.pem" ec2-user@your-ec2-public-dns
     ```

3. **Set Up the Environment**

   ```bash
   # Update the system
   sudo yum update -y

   # Install Python 3 and other dependencies
   sudo yum install -y python3 git postgresql postgresql-devel

   # Clone your project repository
   git clone https://github.com/your-repo/rag-chatbot-app.git
   cd rag-chatbot-app/backend

   # Create a virtual environment and install dependencies
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### 2. Setting Up PostgreSQL with `pgvector` on Amazon RDS

1. **Create an RDS Instance**

   - Go to the RDS Dashboard.
   - Click "Create database".
   - Choose a database creation method (Standard Create).
   - Choose an engine (PostgreSQL).
   - Configure the database (instance specifications, storage, etc.).
   - Set up database credentials.
   - Configure network settings (ensure it can communicate with your EC2 instance).
   - Launch the database instance.

2. **Connect to the RDS Instance**

   - Connect to your PostgreSQL database from your EC2 instance:

     ```bash
     psql --host=your-rds-endpoint --port=5432 --username=your-username --dbname=your-database
     ```

3. **Enable `pgvector` Extension**

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Create the Table with Vector Column**

   ```sql
   CREATE TABLE documents (
       id SERIAL PRIMARY KEY,
       title TEXT,
       text TEXT,
       embedding vector(768)
   );
   ```

### 3. Integrate the Backend with RDS and `pgvector`

1. **Update Your `rag_model.py` to Store and Query Vectors**

   ```python
   import psycopg2
   import torch
   from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

   # Connect to PostgreSQL
   conn = psycopg2.connect(
       host="your-rds-endpoint",
       database="chatbot_db",
       user="your-username",
       password="your-password"
   )
   cursor = conn.cursor()

   # Initialize the tokenizer and model
   tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
   model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

   # Function to add document to the database
   def add_document_to_db(document):
       inputs = tokenizer(document['text'], return_tensors="pt")
       with torch.no_grad():
           embedding = model.retrieval_embeddings(**inputs).detach().numpy().flatten()
       cursor.execute(
           "INSERT INTO documents (title, text, embedding) VALUES (%s, %s, %s)",
           (document['title'], document['text'], embedding)
       )
       conn.commit()

   # Function to query the most similar documents
   def query_similar_documents(query, top_k=5):
       inputs = tokenizer(query, return_tensors="pt")
       with torch.no_grad():
           query_embedding = model.retrieval_embeddings(**inputs).detach().numpy().flatten()
       cursor.execute(
           "SELECT id, title, text, 1 - (embedding <=> %s::vector) AS similarity FROM documents ORDER BY similarity DESC LIMIT %s",
           (query_embedding, top_k)
       )
       return cursor.fetchall()

   # Function to generate response
   def generate_response(query):
       similar_docs = query_similar_documents(query)
       context = " ".join([doc[2] for doc in similar_docs])
       input_ids = tokenizer(query + context, return_tensors="pt").input_ids
       outputs = model.generate(input_ids=input_ids)
       response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
       return response
   ```

2. **Update Your Flask API**

   ```python
   from flask import Flask, request
   from flask_restful import Resource, Api
   from flask_cors import CORS
   import psycopg2
   import redis
   from langchain import LLMChain
   from langchain.chains import SimpleSequentialChain
   from langchain.prompts import ChatPromptTemplate
   from rag_model import generate_response, add_document_to_db

   app = Flask(__name__)
   CORS(app)
   api = Api(app)

   # Connect to Redis
   redis_client = redis.Redis(host='your-redis-endpoint', port=6379, db=0)

   # Initialize LangChain
   prompt_template = ChatPromptTemplate.from_template("The user said: {user_message}\nThe bot should respond with:")
   chain = SimpleSequentialChain([LLMChain(prompt_template=prompt_template)])

   class Message(Resource):
       def get(self):
           cursor.execute("SELECT * FROM messages")
           messages = cursor.fetchall()
           return {'messages': messages}, 200

       def post(self):
           user_message = request.json['message']
           cursor.execute("INSERT INTO messages (message) VALUES (%s)", (user_message,))
           conn.commit()

           # Check cache first
           cached_response = redis_client.get(user_message)
           if cached_response:
               bot_response = cached_response.decode('utf-8')
           else:
               bot_response = generate_response(user_message)
               redis_client.set(user_message, bot_response)

           cursor.execute("INSERT INTO messages (message) VALUES (%s)", (bot_response,))
           conn.commit()

           return {'message': bot_response}, 201

   class Document(Resource):
       def post(self):
           document = request.json['document']
           add_document_to_db(document)
           return {'message': 'Document added'}, 201

   api.add_resource(Message, '/message')
   api.add_resource(Document, '/document')

   if __name__ == '__main__':
       app.run(debug=True)
   ```

### 4. Setting Up Caching with Amazon ElastiCache

1. **Create an ElastiCache Cluster**

   - Go to the ElastiCache Dashboard.
   - Click "Create".
   - Choose a cluster engine (Redis).
   - Configure the cluster (node type, number of nodes, etc.).
   - Set up security groups to allow communication with your EC2 instance.
   - Launch the cluster.

2. **Connect Your Application to ElastiCache**

   Update your Redis connection settings in `app.py` to use the ElastiCache endpoint:

   ```python
   # Connect to Redis
   redis_client = redis.Redis(host='your-elasticache-endpoint', port=6379, db=0)
   ```

### 5. Storing Static Assets with Amazon S3

1. **Create an S3 Bucket**

   - Go to the S3 Dashboard.
   - Click "Create bucket".
   - Name your bucket and configure settings.
   - Create the bucket.

2. **Upload Assets to S3**

   - Use the S3 console to upload files, or use the AWS CLI:

     ```bash
     aws s3 cp your-local-file s3://your-bucket-name/your-object-key
     ```

3. **Serve Static Assets from S3**

   Update your frontend application to load assets from S3.

### 6. Using AWS Lambda for Serverless Functions

1. **Create a Lambda Function**

   - Go to the Lambda Dashboard.
   - Click "Create function".
   - Choose "Author from scratch".
   - Configure function settings.
   - Create the function.

2. **Write the Lambda Function Code**

   Use the Lambda console or upload your code as a .zip file.

3. **Invoke the Lambda Function from Your Application**

   Use the AWS SDK to invoke Lambda functions from your backend or frontend.

### 7. Setting Up API Gateway

1. **Create an API Gateway**

   - Go to the API Gateway Dashboard.
   - Click "Create API".
   - Choose "REST API" or "HTTP API".
   - Configure the API.

2. **Create and Deploy Endpoints**

   Create resources and methods, and deploy the API to a stage.

3. **Update Your Application to Use API Gateway Endpoints**

   Update your frontend and backend to use the API Gateway URLs.

### Example: Full Application Code

#### backend/app.py

```python
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import psycopg2
import redis
from langchain import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from rag_model import generate_response, add_document_to_db

app = Flask(__name

__)
CORS(app)
api = Api(app)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="your-rds-endpoint",
    database="chatbot_db",
    user="your-username",
    password="your-password"
)
cursor = conn.cursor()

# Connect to Redis
redis_client = redis.Redis(host='your-elasticache-endpoint', port=6379, db=0)

# Initialize LangChain
prompt_template = ChatPromptTemplate.from_template("The user said: {user_message}\nThe bot should respond with:")
chain = SimpleSequentialChain([LLMChain(prompt_template=prompt_template)])

class Message(Resource):
    def get(self):
        cursor.execute("SELECT * FROM messages")
        messages = cursor.fetchall()
        return {'messages': messages}, 200

    def post(self):
        user_message = request.json['message']
        cursor.execute("INSERT INTO messages (message) VALUES (%s)", (user_message,))
        conn.commit()

        # Check cache first
        cached_response = redis_client.get(user_message)
        if cached_response:
            bot_response = cached_response.decode('utf-8')
        else:
            bot_response = generate_response(user_message)
            redis_client.set(user_message, bot_response)

        cursor.execute("INSERT INTO messages (message) VALUES (%s)", (bot_response,))
        conn.commit()

        return {'message': bot_response}, 201

class Document(Resource):
    def post(self):
        document = request.json['document']
        add_document_to_db(document)
        return {'message': 'Document added'}, 201

api.add_resource(Message, '/message')
api.add_resource(Document, '/document')

if __name__ == '__main__':
    app.run(debug=True)
```

#### backend/rag_model.py

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
import faiss
import numpy as np
import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="your-rds-endpoint",
    database="chatbot_db",
    user="your-username",
    password="your-password"
)
cursor = conn.cursor()

# Initialize the tokenizer and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Function to add document to the database
def add_document_to_db(document):
    inputs = tokenizer(document['text'], return_tensors="pt")
    with torch.no_grad():
        embedding = model.retrieval_embeddings(**inputs).detach().numpy().flatten()
    cursor.execute(
        "INSERT INTO documents (title, text, embedding) VALUES (%s, %s, %s)",
        (document['title'], document['text'], embedding)
    )
    conn.commit()

# Function to query the most similar documents
def query_similar_documents(query, top_k=5):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.retrieval_embeddings(**inputs).detach().numpy().flatten()
    cursor.execute(
        "SELECT id, title, text, 1 - (embedding <=> %s::vector) AS similarity FROM documents ORDER BY similarity DESC LIMIT %s",
        (query_embedding, top_k)
    )
    return cursor.fetchall()

# Function to generate response
def generate_response(query):
    similar_docs = query_similar_documents(query)
    context = " ".join([doc[2] for doc in similar_docs])
    input_ids = tokenizer(query + context, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response
```

#### frontend/src/App.js

```jsx
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

### Conclusion for Discovery 3

By following this tutorial, you can create a scalable chatbot application using AWS services. This setup includes hosting the backend on EC2, managing a PostgreSQL database with RDS (enhanced with `pgvector` for vector search), implementing caching with ElastiCache, storing static assets on S3, and optionally using Lambda for serverless functions and API Gateway for managing APIs. This approach ensures a robust, scalable, and efficient chatbot application following industry best practices.

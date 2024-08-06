# Building a Full-Stack Chatbot with LangChain and Custom RAG Model

This tutorial will guide you through creating a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application will have a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base.

## Project Structure

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

### Step 1: Backend Setup

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

### Step 2: Frontend Setup

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

### Step 3: Integrate Custom Material for RAG

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

### Step 4: Run and Test the Application

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

### Conclusion

By following this tutorial, you have created a full-stack chatbot application that leverages LangChain and a custom Retrieval-Augmented Generation (RAG) model. The application includes a backend built with Flask, a frontend built with React, and a custom RAG model that enhances the chatbot's responses using your own knowledge base. LangChain helps manage continuous dialog effectively, providing a seamless conversational experience

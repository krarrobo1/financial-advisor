# Financial Advisor Chatbot

## Culmination project of the ML Engineering program at  [Anyone AI](https://www.linkedin.com/school/anyone-ai/).

### Team members:

+ [Sergio Prieto](https://www.linkedin.com/in/serprieto/)

+ [Ricardo Arrobo](https://www.linkedin.com/in/krarroboc)

+ [Martin Lana](https://www.linkedin.com/in/mart%C3%ADn-ignacio-lana-bengut/)

+ [Carlos Gómez](https://www.linkedin.com/in/carlosgomez88)

+ [Manuel Tablado](https://www.linkedin.com/in/manuel-tablado/)

### Mentor:
+ [Claudio A. Gauna](https://www.linkedin.com/in/claudio-andres-gauna-2b697b97/)

# OVERVIEW
Within this initiative, we've developed a chatbot tailored to provide investment insights, focusing primarily on public corporations from the NASDAQ stock exchange. The bot employs a conversational mechanism to process thoughts using the ChatGPT API. Further, it taps into a comprehensive database encompassing approximately 10,000 financial records and online news sources, ensuring that users receive credible and precise feedback.
# Objective
The core objective of this initiative is to furnish users with a platform where they can converse with an AI chatbot and inquire about financial aspects concerning NASDAQ listed firms.
### Key Steps:

- Dataset Exploratory Analysis(EDA)

- Preprocessing and data structuring

- Cataloging textual documents in the database

- Creating a system for processing user queries and formulating responses

- Setting up an API (FastAPI) for bot interactions

- Crafting a user interface reminiscent of ChatGPT(Streamlit)

- System demo presentation

- Dockerization for portability 

# Architecture
In order to enable modular development, scalability, and flexibilitywe've adopted a microservices-oriented architecture. Each service is containerized using Docker, and the orchestration of these containers is managed by docker-compose:


![image](https://github.com/martinlanabengut/AngularProyect-MarketOnline/assets/53227496/e4fdd759-330b-4793-ad09-35dc042a3646)


## Project structure


# ETL
The ETL process leverages both boto3 and Airflow. Boto3 is employed to retrieve documents from an AWS bucket, while Airflow orchestrates and schedules the ETL workflows. Documents, fetched in PDF format, are classified on a company-wise basis.

# API
Our system integrates FastAPI for API development, enabling seamless interaction between the frontend and the AI-backed response generator.

# Generative-retriever model
The generative-retriever model is referred to as the code that receives the user query, retrieves relevant information, and generates an appropriate answer for the user.
### Agent
The core model is essentially the Agent of [LangChain](https://python.langchain.com/en/latest/index.html). This Agent is a highly versatile, prompt-based component harnessing a large language model (LLM) to provide reasoning-based answers, surpassing the limits of traditional extractive or generative question-answering systems.
### Tools
Our agent is equipped with two distinct tools:

1. A **Retrieval Question/Answering (QA)** tool, purpose-built for answering questions related to the stored documents. This tool accepts the **Input Action** from the agent, then conducts a search through the most relevant document fragments archived in our ChromaDB vector database.

2. A tool integrated with the **DuckDuckGo search engine**, designed to be deployed if the Retrieval QA tool, using our vector database, cannot provide a satisfactory answer.

Furthermore, we've made alterations to the agent's prompt to ensure the primary utilization of the Retrieval QA tool. The agent will only resort to DuckDuckGo after it has first attempted to address the query using the vector database.

From the observation provided by the tools, the agent forms a new thought process, determining whether to use the tool once more or to deliver a direct response to the user's query. 


# User Interface
The user interface is crafted using Streamlit, offering an intuitive and interactive experience. To ensure smooth performance and optimal rendering, the interface elements are modularized into individual Streamlit components. Streamlit's Python-centric approach allows for rapid development and deployment, while its built-in widgets and styling options ensure a visually appealing and user-friendly interface.

![WhatsApp Image 2023-10-13 at 21](https://github.com/martinlanabengut/AngularProyect-MarketOnline/assets/53227496/5760c68a-747f-4711-a83f-9c2bdd41e723)



## **Installation Guide**

### **Prerequisites:**
1. **Docker Desktop Installation:**  
   Ensure you have Docker Desktop installed on your system. If not, you can download and install it from [here](https://www.docker.com/products/docker-desktop).

### **ETL Installation Steps:**
1. **Set Environment Variables:**  
   Ensure you've set up necessary environment variables, especially the AWS_ACCESS_KEY and AWS_SECRET_KEY provided in the project assets.
   ```
   cp .env.example .env
   ```

2. **Docker and Docker-Compose Installation:**  
   If you haven’t already installed Docker and Docker-Compose from the prerequisites, ensure you do so now.

3. **Build and Start ETL Containers:**  
   Execute the following command to download the images, build them, and start the containers. This may take a few minutes.
   ```
   docker-compose up . -d
   ```

4. **Access the Airflow Interface:**  
   Navigate to [http://localhost:8080](http://localhost:8080) on your web browser.

5. **Login to Airflow:**  
   Use the default credentials:
   - **Username:** `airflow`
   - **Password:** `airflow`

6. **Execute the ETL Taskflow:**  
   Manually start the `elt_taskflow` dag by clicking the play button.

7. **Monitor the DAG:**  
   Allow the dag to complete its tasks.

8. **Access the Vector DB:**  
   Once completed, you can check the `dataset/chroma_db-database` to access the Vector DB.

### **UI & API Installation Steps:**
1. **Build and Launch the Containers:**  
   Run the following command in your terminal or command line to build and launch the Docker containers:
   ```
   docker-compose up --build -d
   ```

2. **Access the UI Container:**  
   After the containers are up and running, you can access the User Interface (UI) container through your web browser.

    - UI: [http://localhost:8501/](http://localhost:8501/)
    - API: [http://localhost:8000/](http://localhost:8000/)


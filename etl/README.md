
# ETL Pipeline

# Extract

- Download PDF Dataset from AWS

# Transform and Load

- Create embeddings using the transformer

- Load embeddings to ChromaDB

  

# Steps to run

1. Set your environment variables, make sure including AWS_ACCESS_KEY and AWS_SECRET_KEY given in the project assets.

```

cp .env.example .env

```

2. Install [docker](https://docs.docker.com/engine/install/) and [docker compose](https://docs.docker.com/compose/install/)

3. Run the following docker compose command and wait until the images are downloaded, builded and containers started (this may take several minutes)

```

docker-compose up . -d

```

4. Go to http://localhost:8080

5. Login using the default credentials (username: airflow, password: airflow)

6. Start the `elt_taskflow` dag manually by clicking the play button

7. Wait until the dag ends their tasks

8. Check dataset/chroma_db-database to access to the Vector DB
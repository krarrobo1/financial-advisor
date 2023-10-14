FROM apache/airflow:latest-python3.10
ENV HNSWLIB_NO_NATIVE=1
ADD requirements.txt . 
ADD .env .
RUN pip install -r requirements.txt 
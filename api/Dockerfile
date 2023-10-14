FROM python:3.11
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
RUN mkdir -p db
ENTRYPOINT ["uvicorn", "--workers=1", "--host", "0.0.0.0", "api_cr_chain_hist:app"]
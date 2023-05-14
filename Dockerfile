# Install env 
FROM python:3.9 
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace/ 
RUN pip install -r requirements.txt 

# Copy lib 
COPY pyMSOO /workspace/pyMSOO
# COPY run.sh /workspace/run.sh 

# CMD bash run.sh 
CMD "ok"
# CMD echo 'helloworld'
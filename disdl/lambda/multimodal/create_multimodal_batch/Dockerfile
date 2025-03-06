FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ./
RUN python3.10 -m pip install -r requirements.txt -t .

COPY app.py ./
COPY bert-tokenizer /var/task/bert-tokenizer
#COPY model /opt/ml/model
CMD ["app.lambda_handler"]

# # Use a newer Python version if needed
# FROM public.ecr.aws/lambda/python:3.11.2023.12.05.16-x86_64
# # 
# # Install Python dependencies
# COPY requirements.txt ./
# RUN python3.11 -m pip install -r requirements.txt -t .

# # Copy application code
# COPY app.py ./

# # Define the Lambda handler
# CMD ["app.lambda_handler"]

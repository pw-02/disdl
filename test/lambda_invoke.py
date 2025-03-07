import json
import boto3

class LambdaInvoker:
    def __init__(self, lambda_name):
        # Initialize Lambda client and set Lambda function name
        self.lambda_client = boto3.client('lambda')
        self.lambda_name = lambda_name

    def invoke_lambda(self):
        # Define the payload to pass to the Lambda function
        payload = {
            "bucket_name": "sdl-cifar10",
            "batch_id": "1",
            "cache_address": "10.0.25.0:6378",
            "task": "prefetch",
            "batch_samples": [
                ["train/Airplane/aeroplane_s_000004.png", 0]
            ]
        }

        # Invoke the Lambda function asynchronously
        response = self.lambda_client.invoke(
            FunctionName=self.lambda_name,
            InvocationType="Event",  # Asynchronous invocation
            Payload=json.dumps(payload)  # Pass the payload as a JSON string
        )

        # Optionally, print the response or handle it
        print(f"Lambda invoked, response status code: {response['StatusCode']}")

# Example usage:
lambda_name = "CreateVisionTrainingBatch"  # Replace with your Lambda function name
invoker = LambdaInvoker(lambda_name)
invoker.invoke_lambda()

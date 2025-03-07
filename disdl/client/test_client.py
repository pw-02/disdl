
from disdl_client import DisDLClient
if __name__ == "__main__":
    num_epochs = 2
    client = DisDLClient("localhost:50051")
    jobid, dataset_info = client.registerJob("s3://sdl-cifar10/test/", "cifar10")
    print(f"Job ID: {jobid}, Dataset Info: {dataset_info}")

    num_batchs = dataset_info["num_batches"]
    for j in range(num_epochs):
        for i in range(num_batchs):
            batch_id, samples, is_cached = client.sampleNetMinibatch()
            print(f"Batch ID: {batch_id}, Num Samples: {len(samples)}, Is Cached: {is_cached}")
    client.sendJobEndNotification()
    client.close()
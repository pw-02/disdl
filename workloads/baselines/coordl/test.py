import torch
from torch.utils.data import IterableDataset, DataLoader

# Step 1: Create a custom IterableDataset
class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        super(MyIterableDataset, self).__init__()
        self.data = data  # Your dataset (could be a list, numpy array, etc.)

    def __iter__(self):
        # Define how to iterate over your data
        for item in self.data:
            yield item  # Yield items one by one
if __name__ == '__main__':
    # Example data
    data = [i for i in range(100)]  # A list of data (this could be large, such as file paths)

    # Step 2: Instantiate your dataset
    dataset = MyIterableDataset(data)

    # Step 3: Create a DataLoader with multiple workers
    data_loader = DataLoader(dataset, batch_size=10, num_workers=4)

    # Step 4: Iterate over the DataLoader
    for batch in data_loader:
        print(batch)

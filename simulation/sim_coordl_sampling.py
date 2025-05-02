from collections import defaultdict

# Config
total_samples = 9000
batch_size = 100
num_dls = 3

# Derived
total_batches = total_samples // batch_size
batches_per_dl = total_batches // num_dls

# Each DL owns a disjoint set of batches
dl_owns = {}
for dl_id in range(num_dls):
    start = dl_id * batches_per_dl
    end = start + batches_per_dl
    dl_owns[dl_id] = list(range(start, end))

# Build global list of batches in round-robin order across DLs
global_batch_sequence = []
for i in range(batches_per_dl):
    for dl_id in range(num_dls):
        if i < len(dl_owns[dl_id]):
            global_batch_sequence.append(dl_owns[dl_id][i])

# Now simulate each DL consuming the full sequence
dl_consumes = defaultdict(list)
for dl_id in range(num_dls):
    for step, batch_id in enumerate(global_batch_sequence):
        # Find out who produced it
        producer = next((owner for owner, batches in dl_owns.items() if batch_id in batches), None)
        dl_consumes[dl_id].append((step, batch_id, producer))

# Print out the consumption trace
for dl_id in range(num_dls):
    print(f"\nDL{dl_id} will consume batches in this order:")
    for step, batch_id, producer in dl_consumes[dl_id]:
        print(f"  Step {step:2d}: Batch {batch_id:2d} (from DL{producer})")

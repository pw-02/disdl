from collections import defaultdict

# Simulation settings
total_samples = 9000
batch_size = 100
total_dls = 3

# Derived values
total_batches = total_samples // batch_size
batches_per_dl = total_batches // total_dls

# Assign batches to producer DLs
dl_owns = {}
for dl in range(total_dls):
    start = dl * batches_per_dl
    end = start + batches_per_dl
    dl_owns[dl] = list(range(start, end))

# Create per-DL consumption logs
dl_consumes = defaultdict(list)

# For each DL (job), simulate consuming all batches
for dl in range(total_dls):
    producer_ptrs = [0] * total_dls  # Track how far each producer is along
    step = 0
    while len(dl_consumes[dl]) < total_batches:
        producer = step % total_dls
        if producer_ptrs[producer] < len(dl_owns[producer]):
            batch_id = dl_owns[producer][producer_ptrs[producer]]
            dl_consumes[dl].append((step, batch_id, producer))
            producer_ptrs[producer] += 1
        step += 1

# Print results
for dl in range(total_dls):
    print(f"\nDL{dl} will consume {len(dl_consumes[dl])} batches:")
    for step, batch_id, producer in dl_consumes[dl]:
        print(f"  Step {step:2d}: Batch {batch_id:2d} (from DL{producer})")

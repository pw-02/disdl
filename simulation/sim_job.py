class Job:
    def __init__(self, job_id: str, speed: float):
        self.job_id = job_id
        self.speed = speed
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.elapased_time_sec = 0
        self.num_batches_processed = 0
        self.local_cache = {} #used by tensorocket producer to store batches
    
    def perf_stats(self, horurly_ec2_cost=12.24, hourly_cache_cost=3.25):
        hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
        throughput = self.num_batches_processed / self.elapased_time_sec if self.elapased_time_sec > 0 else 0
        self.compute_cost = (horurly_ec2_cost / 3600) * self.elapased_time_sec
        cache_cost = (hourly_cache_cost / 3600) * self.elapased_time_sec
        total_cost = self.compute_cost + hourly_cache_cost
        return {
            'job_id': self.job_id,
            'job_speed': self.speed,
            'batches_processed': self.num_batches_processed,
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'cache_hit_%': hit_rate,
            'elapsed_time': self.elapased_time_sec,
            'throughput(batches/s)': throughput,
            'compute_cost': self.compute_cost,
            'cache_cost': cache_cost,
            'total_cost': total_cost
            }
    def __lt__(self, other):
        return self.speed < other.speed  # Compare based on speed

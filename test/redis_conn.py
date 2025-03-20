#quizk get with redis con
import redis

host = '54.200.19.61'
port = 6378

key = f'{88}file1'
r = redis.StrictRedis(host=host, port=port)

#put
# memory_file = b"\0" * (8 * 1024 * 1024)  # 8mb
# r.set(key, memory_file)

print(r.get(key))

import redis
import time
import io


host = '54.200.19.61'
port = 6378

#add some data to redis and then check if it is there every 30 seconds and print it out. When it is not there, print out that it is not there
#print the total time it was in redis. 

# keys = list(range(170))

def put(keys):
    r = redis.StrictRedis(host=host, port=port) 
    for _ in keys:
        try:
            memory_file = io.BytesIO(b"\0" * (8 * 1024 * 1024))  # 8mb
            r.set(f'{_}file', memory_file.getvalue())
            print(r.get(f'{_}file'))
         
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(15)
def get(keys):
    r = redis.StrictRedis(host=host, port=port) 
    started = time.perf_counter()
    sleep_time = 30
    while True:
        for key in keys:
            try:
                response = r.get(f'{key}file')
                if response == None:
                    break
                else:
                    print(f"Response: {response}. Total time: {time.perf_counter() - started}")
                time.sleep(sleep_time)
                sleep_time += 30
            except Exception as e:
                print(f"Error: {e}")
                #time.sleep(15)
        break
    print(f"Evicted Total time: {time.perf_counter() - started}")

keys = list(range(1, 340))
put(keys)
get(keys)

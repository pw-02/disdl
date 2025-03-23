import redis
import time
import io


host = '35.166.232.0'
port = 6378

#add some data to redis and then check if it is there every 30 seconds and print it out. When it is not there, print out that it is not there
#print the total time it was in redis. 

# keys = list(range(170))

def put(keys):
    r = redis.StrictRedis(host=host, port=port)
    toal_mb = 0
    for _ in keys:
        try:
            memory_file = io.BytesIO(b"\0" * (8 * 1024 * 1024))  # 8mb
            toal_mb += 8
            r.set(f'{_}', memory_file.getvalue())
            print(f"Put: {_}, Total MB: {toal_mb}")
            time.sleep(5)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
def get(keys):
    r = redis.StrictRedis(host=host, port=port) 
    started = time.perf_counter()
    sleep_time = 1
    while True:
        for key in keys:
            try:
                response = r.get(f'{key}')
                if response == None:
                    break
                else:
                    print(f"Response: {key}. Total time: {time.perf_counter() - started}")
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

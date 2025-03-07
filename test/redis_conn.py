import redis
host = '54.69.247.89'
port = 6378

r = redis.StrictRedis(host=host, port=port)
# r.set('foo', 'bar')
print(r.get('0_0_19_29c6bad2c6762388'))
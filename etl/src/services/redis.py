from src.settings import REDIS_IP, REDIS_DB_ID, REDIS_PORT, PROCESSED_KEY
import redis
import logging

r = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=REDIS_DB_ID)


def save_docs(key: str, data: list):
    try:
        r.rpush(key, *data)
        logging.info(f"List saved to Redis with key '{key}'")
    except Exception as e:
        logging.info(f"Error saving list to Redis: {str(e)}")


def retrieve_docs(key: str):
    try:
        # Retrieve the list from Redis using the LRANGE command
        result = r.lrange(key, 0, -1)
        # Convert the bytes returned by Redis to a list of strings
        decoded_result = [item.decode() for item in result]

        return decoded_result

    except Exception as e:
        logging(f"Error retrieving list from Redis: {str(e)}")
        return []


def save_processed_path_to_redis(path):
    try:
        r.sadd(PROCESSED_KEY, path)
        print(
            f"Path '{path}' has been saved as processed in Redis with key '{PROCESSED_KEY}'"
        )
    except Exception as e:
        print(f"Error saving path to Redis: {str(e)}")


def check_path_exists_in_redis(path):
    try:
        # Check if the path exists in the Redis set using the SISMEMBER command
        exists = r.sismember(PROCESSED_KEY, path)
        return exists
    except Exception as e:
        print(f"Error checking path existence in Redis: {str(e)}")
        return False

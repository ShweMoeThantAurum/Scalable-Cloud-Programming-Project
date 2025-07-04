import boto3
import json
import time
from collections import defaultdict, deque
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS configuration
kinesis_client = boto3.client('kinesis', region_name='us-east-1')
s3_client = boto3.client('s3')
stream_name = 'electronics-reviews-stream'
bucket_name = 'electronic-reviews-bucket'
output_prefix = 'stream_output/'

def get_shard_iterator(stream_name, shard_id):
    """Get shard iterator for Kinesis stream."""
    try:
        response = kinesis_client.get_shard_iterator(
            StreamName=stream_name,
            ShardId=shard_id,
            ShardIteratorType='TRIM_HORIZON'
        )
        return response['ShardIterator']
    except Exception as e:
        logger.error(f"Error getting shard iterator: {e}")
        raise

def process_records(records, word_counts, window_start, window_duration=300, slide_interval=60):
    """Process records, filter keywords, and update sliding window."""
    current_time = time.time()
    new_records = []
    for record in records:
        try:
            data = json.loads(record['Data'])
            text = data['cleaned_reviewText']
            send_time = data['send_time']
            # Filter for keywords
            if 'great' in text.lower() or 'poor' in text.lower():
                words = [word for word in text.lower().split() if word not in ('', ' ', 'a', 'the', 'and', 'is')]
                for word in words:
                    word_counts.append((word, send_time))
                new_records.append(data)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in record: {record['Data']}")

    # Remove old records from the front of the deque
    while word_counts and current_time - word_counts[0][1] > window_duration:
        word_counts.popleft()

    # Count words in current window
    word_freq = defaultdict(int)
    for word, _ in word_counts:
        word_freq[word] += 1

    # Get top 5 words
    top_5 = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return top_5, current_time, new_records


def save_to_s3(top_5, window_time, new_records):
    """Save top 5 words and filtered records to S3."""
    output_key = f"{output_prefix}top_5_words_{int(window_time)}.json"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=output_key,
        Body=json.dumps({
            'window_time': window_time,
            'top_5': top_5,
            'filtered_records': new_records
        })
    )
    logger.info(f"Saved top 5 to s3://{bucket_name}/{output_key}")

def main():
    # Get shard IDs
    try:
        response = kinesis_client.describe_stream(StreamName=stream_name)
        shard_id = response['StreamDescription']['Shards'][0]['ShardId']
        shard_iterator = get_shard_iterator(stream_name, shard_id)
    except Exception as e:
        logger.error(f"Error describing stream: {e}")
        return

    # Sliding window state
    word_counts = deque()  # Stores (word, timestamp) pairs
    window_duration = 300  # 5 minutes in seconds
    slide_interval = 60    # 1 minute in seconds
    last_slide = time.time()

    try:
        while True:
            response = kinesis_client.get_records(ShardIterator=shard_iterator, Limit=100)
            records = response['Records']
            if records:
                top_5, current_time, new_records = process_records(records, word_counts, last_slide, window_duration, slide_interval)
                if current_time - last_slide >= slide_interval:
                    save_to_s3(top_5, current_time, new_records)
                    last_slide = current_time
                    logger.info(f"Top 5 words: {top_5}")
            shard_iterator = response['NextShardIterator']
            time.sleep(0.1)  # Avoid excessive API calls

    except Exception as e:
        logger.error(f"Error during stream processing: {e}")
        raise

if __name__ == "__main__":
    main()
import boto3
import json
import time
from collections import defaultdict, deque
import logging
from textblob import TextBlob

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
            ShardIteratorType='LATEST'
        )
        return response['ShardIterator']
    except Exception as e:
        logger.error(f"Error getting shard iterator: {e}")
        raise

def process_records(records, word_counts, window_duration=300):
    """Process records, filter with enhanced keywords and sentiment, update sliding window."""
    current_time = time.time()
    new_records = []
    keywords = ["great", "poor", "excellent", "bad", "quality", "awesome", "terrible"]
    
    for record in records:
        try:
            data = json.loads(record['Data'])
            text = data['cleaned_reviewText']
            send_time = data['send_time']
            # Check for keywords
            if any(keyword in text.lower() for keyword in keywords):
                # Sentiment analysis
                blob = TextBlob(text)
                polarity = float(blob.sentiment.polarity)
                category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral'
                # Tokenize and filter words
                words = [word for word in text.lower().split() if word not in ('', ' ', 'a', 'the', 'and', 'is')]
                for word in words:
                    word_counts.append((word, send_time, category))
                new_records.append({
                    'cleaned_reviewText': text,
                    'send_time': send_time,
                    'sentiment_category': category,
                    'polarity': polarity
                })
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in record: {record['Data']}")

    # Remove old records from deque
    while word_counts and current_time - word_counts[0][1] > window_duration:
        word_counts.popleft()

    # Count words in current window, grouped by sentiment
    word_freq = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
    for word, _, category in word_counts:
        word_freq[word][category] += 1
        word_freq[word]['total'] += 1

    # Get top 5 words overall
    top_5 = sorted(
        [(word, counts['total']) for word, counts in word_freq.items()],
        key=lambda x: x[1], reverse=True
    )[:5]

    # Include sentiment breakdown for top 5
    top_5_with_sentiment = [
        (word, count, {
            'positive': word_freq[word]['positive'],
            'negative': word_freq[word]['negative'],
            'neutral': word_freq[word]['neutral']
        }) for word, count in top_5
    ]
    return top_5_with_sentiment, current_time, new_records

def save_to_s3(top_5, window_time, new_records):
    """Save top 5 words with sentiment breakdown and filtered records to S3."""
    output_key = f"{output_prefix}top_5_words_{int(window_time)}.json"
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=json.dumps({
                'window_time': window_time,
                'top_5_with_sentiment': [
                    {'word': word, 'count': count, 'sentiment_breakdown': sentiment}
                    for word, count, sentiment in top_5
                ],
                'filtered_records': new_records
            })
        )
        logger.info(f"Saved top 5 to s3://{bucket_name}/{output_key}")
    except Exception as e:
        logger.error(f"Error saving to S3: {e}")

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
    word_counts = deque()  # Stores (word, timestamp, sentiment)
    window_duration = 300  # 5 minutes
    slide_interval = 60    # 1 minute
    last_slide = time.time()

    try:
        while True:
            response = kinesis_client.get_records(ShardIterator=shard_iterator, Limit=100)
            records = response['Records']
            if records:
                start_time = time.time()
                top_5_with_sentiment, current_time, new_records = process_records(records, word_counts, window_duration)
                if current_time - last_slide >= slide_interval:
                    save_to_s3(top_5_with_sentiment, current_time, new_records)
                    last_slide = current_time
                    # Only log word and count
                    simplified_top_5 = [(word, count) for word, count, _ in top_5_with_sentiment]
                    logger.info(f"Top 5 words: {simplified_top_5}")
            shard_iterator = response['NextShardIterator']
            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Error during stream processing: {e}")
        raise

if __name__ == "__main__":
    main()

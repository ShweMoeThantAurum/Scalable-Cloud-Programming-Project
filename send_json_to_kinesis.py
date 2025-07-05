import boto3
import json
import time
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS and Kinesis configuration
s3_client = boto3.client('s3')
kinesis_client = boto3.client('kinesis', region_name='us-east-1')
stream_name = 'electronics-reviews-stream'
bucket_name = 'electronic-reviews-bucket'
input_prefix = 'processed_text_data_json/'

def send_batch_to_kinesis(records, max_retries=3):
    """Send a batch of records to Kinesis with retries."""
    for attempt in range(max_retries):
        try:
            response = kinesis_client.put_records(
                StreamName=stream_name,
                Records=[
                    {
                        'Data': json.dumps({
                            'cleaned_reviewText': record['cleaned_reviewText'],
                            'send_time': time.time()
                        }),
                        'PartitionKey': str(random.randint(0, 1000))
                    } for record in records
                ]
            )
            logger.info(f"Sent {len(records)} records to Kinesis: {response}")
            if response['FailedRecordCount'] == 0:
                return response
            else:
                logger.warning(f"Failed to send {response['FailedRecordCount']} records")
                # Optionally, retry only failed records
        except Exception as e:
            logger.error(f"Error sending to Kinesis (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    logger.error(f"Failed to send batch after {max_retries} attempts")
    return None

def main():
    try:
        # List JSON files in S3
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)
        json_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]

        if not json_files:
            logger.error("No JSON files found in %s", input_prefix)
            return

        # Process each JSON file
        for json_key in json_files:
            logger.info(f"Processing {json_key}")
            # Download JSON file
            obj = s3_client.get_object(Bucket=bucket_name, Key=json_key)
            json_data = obj['Body'].read().decode('utf-8').splitlines()

            # Parse JSON lines
            records = [json.loads(line) for line in json_data] 
            batch_size = 100 
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                send_batch_to_kinesis(batch)
                time.sleep(0.1)
                logger.info(f"Processed batch {i // batch_size + 1}")

        logger.info("Finished sending all records to Kinesis")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise

if __name__ == "__main__":
    main()

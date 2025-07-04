import boto3
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS and Kinesis configuration
s3_client = boto3.client('s3')
kinesis_client = boto3.client('kinesis', region_name='us-east-1')
stream_name = 'electronics-reviews-stream'
bucket_name = 'electronic-reviews-bucket'
input_prefix = 'processed_text_data_json/'

def send_batch_to_kinesis(records):
    """Send a batch of records to Kinesis."""
    try:
        response = kinesis_client.put_records(
            StreamName=stream_name,
            Records=[
                {
                    'Data': json.dumps({'cleaned_reviewText': record['cleaned_reviewText'], 'send_time': time.time()}),
                    'PartitionKey': str(i)
                } for i, record in enumerate(records)
            ]
        )
        logger.info(f"Sent {len(records)} records to Kinesis: {response}")
        if response['FailedRecordCount'] > 0:
            logger.warning(f"Failed to send {response['FailedRecordCount']} records")
        return response
    except Exception as e:
        logger.error(f"Error sending to Kinesis: {e}")
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

            # Parse JSON lines and send in batches
            batch_size = 100  # Kinesis allows up to 500 records per put_records
            records = [json.loads(line) for line in json_data[:1000]]  # Limit to 1000 records per file
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                send_batch_to_kinesis(batch)
                time.sleep(0.1)  # 10 batches/sec (1000 records/s max for 1 shard)
                logger.info(f"Processed batch {i // batch_size + 1}")

        logger.info("Finished sending all records to Kinesis")

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise

if __name__ == "__main__":
    main()
import boto3
import pandas as pd
import pyarrow.parquet as pq
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS configuration
s3_client = boto3.client('s3')
bucket_name = 'electronic-reviews-bucket'
input_prefix = 'processed_text_data_original/'
output_prefix = 'processed_text_data_json/'

def convert_parquet_to_json():
    try:
        # List Parquet files in S3
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=input_prefix)
        parquet_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]

        if not parquet_files:
            logger.error("No Parquet files found in %s", input_prefix)
            return

        # Process each Parquet file
        for parquet_key in parquet_files:
            # Skip already converted files
            json_key = output_prefix + parquet_key.split("/")[-1].replace('.parquet', '.json')
            try:
                s3_client.head_object(Bucket=bucket_name, Key=json_key)
                logger.info(f"Skipping {parquet_key} (already converted to {json_key})")
                continue
            except s3_client.exceptions.ClientError:
                pass  # File doesn't exist, proceed with conversion

            logger.info(f"Processing {parquet_key}")
            # Download Parquet file
            obj = s3_client.get_object(Bucket=bucket_name, Key=parquet_key)
            parquet_file = io.BytesIO(obj['Body'].read())

            # Read Parquet into Pandas DataFrame
            df = pq.read_table(parquet_file).to_pandas()

            # Convert to JSON in memory
            json_buffer = io.StringIO()
            df[['cleaned_reviewText']].to_json(json_buffer, orient='records', lines=True)

            # Upload JSON to S3
            s3_client.put_object(
                Bucket=bucket_name,
                Key=json_key,
                Body=json_buffer.getvalue().encode('utf-8')
            )
            logger.info(f"Uploaded JSON to s3://{bucket_name}/{json_key}")

    except Exception as e:
        logger.error(f"Error converting Parquet to JSON: {e}")
        raise

if __name__ == "__main__":
    convert_parquet_to_json()
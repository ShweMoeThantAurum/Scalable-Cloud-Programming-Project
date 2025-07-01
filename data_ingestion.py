import logging
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, spark_partition_id
from pyspark.sql.types import StructType, StructField, StringType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_PATH = "s3://electronics-reviews-bucket/Electronics.json"
OUTPUT_PATH = "s3://electronics-reviews-bucket/processed_text_data_original/"
BUCKET_NAME = "electronics-reviews-bucket"
OUTPUT_PREFIX = "processed_text_data_original/"

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("ElectronicsTextDataIngestion").getOrCreate()

    # Define schema
    schema = StructType([StructField("reviewText", StringType(), True)])

    try:
        # Read and process data
        df_text = spark.read.option("multiLine", False).schema(schema).json(INPUT_PATH) \
            .filter(col("reviewText").isNotNull()) \
            .select(lower(col("reviewText")).alias("cleaned_reviewText"))

        # Write to S3 in Parquet format
        df_text.write.mode("overwrite").parquet(OUTPUT_PATH)

        # Log metrics
        record_count = df_text.count()
        partition_count = df_text.select(spark_partition_id()).distinct().count()
        logger.info(f"Total records ingested: {record_count}")
        logger.info(f"Number of partitions: {partition_count}")

        # Calculate and log output size
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=OUTPUT_PREFIX)
        total_size_bytes = sum(obj['Size'] for obj in response.get('Contents', []))
        total_size_gb = total_size_bytes / (1024 ** 3)
        logger.info(f"Total output file size: {total_size_gb:.2f} GB")

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()

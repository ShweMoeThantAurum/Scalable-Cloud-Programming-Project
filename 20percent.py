"""Process 20% of a dataset for sentiment analysis and keyword extraction using Spark."""

import json
import logging
import os
import time
from typing import Dict, List, Tuple, Union

import boto3
import pandas as pd
import plotly.graph_objects as go
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, spark_partition_id, trim, udf
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('py4j').setLevel(logging.WARNING)

# Constants
INPUT_PATH = "s3://electronic-reviews-bucket/processed_text_data_original/"
KEYWORD_OUTPUT_PATH = "s3://electronic-reviews-bucket/keyword_results/"
SENTIMENT_OUTPUT_PATH = "s3://electronic-reviews-bucket/sentiment_results/"
BUCKET_NAME = "electronic-reviews-bucket"
KEYWORD_PLOT_PATH = "results/keyword_plots/"
SENTIMENT_PLOT_PATH = "results/sentiment_plots/"
METRICS_PATH = "results/metrics/"
KEYWORD_JSON_PATH = "results/keyword_results/"
SENTIMENT_JSON_PATH = "results/sentiment_results/"


def map_sentiment_udf(text: Union[str, None]):
    """Perform sentiment analysis on a given text using TextBlob."""
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid text input: {text}")
        return (0.0, "neutral")
    try:
        blob = TextBlob(text)
        polarity = float(blob.sentiment.polarity)
        category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral'
        return (polarity, category)
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return (0.0, "neutral")


# Register UDF for sentiment analysis
sentiment_udf = udf(
    map_sentiment_udf,
    StructType([
        StructField("polarity", FloatType(), False),
        StructField("category", StringType(), False)
    ])
)


def upload_to_s3(local_path: str, s3_key: str, max_retries: int = 3):
    """Upload a file to S3 with exponential backoff retries."""
    s3_client = boto3.client('s3')
    for attempt in range(max_retries):
        try:
            s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
            logger.info(f"Uploaded to s3://{BUCKET_NAME}/{s3_key}")
            return
        except Exception as e:
            logger.warning(f"S3 upload attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to upload to s3://{BUCKET_NAME}/{s3_key} after {max_retries} attempts")
                raise
            time.sleep(2 ** attempt)


def plot_top_words(top_words_df: pd.DataFrame, fraction: float):
    """Generate and save a bar chart of the top 10 words with their frequencies."""
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff7f0e']
    fig = go.Figure([go.Bar(
        x=top_words_df['word'],
        y=top_words_df['frequency'],
        text=top_words_df['frequency'],
        textposition='auto',
        marker_color=colors[:len(top_words_df)],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.85
    )])
    # Set dynamic title based on fraction
    title = "Top 10 Words for 20% of the Dataset" if fraction == 0.2 else f"Top 10 Words for {int(fraction*100)}% of the Dataset"
    if fraction == 1.0:
        title = "Top 10 Words for the Whole Dataset"
    fig.update_layout(
        title=title,
        xaxis_title="Word",
        yaxis_title="Frequency",
        title_x=0.5,
        height=600,
        width=800,
        font=dict(family="Arial, sans-serif", size=16, color="black"),
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45, gridcolor='lightgray', title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(gridcolor='lightgray', title_font=dict(size=18), tickfont=dict(size=14)),
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=100)
    )
    # Save plot to temporary file and upload to S3
    path = f"/tmp/top_words_fraction_{fraction:.2f}.png"
    fig.write_image(path, format="png", scale=2)
    upload_to_s3(path, f"{KEYWORD_PLOT_PATH}top_words_fraction_{fraction:.2f}.png")


def plot_sentiment_categories(category_counts: List[Tuple[str, int]], fraction: float):
    """Generate and save a pie chart of sentiment category counts."""
    df = pd.DataFrame(category_counts, columns=['category', 'count'])
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # Colors for positive, neutral, negative
    fig = go.Figure([go.Pie(
        labels=df['category'],
        values=df['count'],
        textinfo='label+percent',
        textposition='inside',
        marker=dict(colors=colors[:len(df)], line=dict(color='black', width=1.5)),
        opacity=0.85
    )])
    title = "Sentiment Analysis for 20% of the Dataset" if fraction == 0.2 else f"Sentiment Analysis for {int(fraction*100)}% of the Dataset"
    if fraction == 1.0:
        title = "Sentiment Analysis for the Whole Dataset"
    fig.update_layout(
        title=title,
        title_x=0.5,
        height=600,
        width=800,
        font=dict(family="Arial, sans-serif", size=16, color="black"),
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(title="Sentiment Categories", font=dict(size=14)),
        margin=dict(l=50, r=50, t=100, b=100)
    )
    path = f"/tmp/categories_fraction_{fraction:.2f}.png"
    fig.write_image(path, format="png", scale=2)
    upload_to_s3(path, f"{SENTIMENT_PLOT_PATH}categories_fraction_{fraction:.2f}.png")


def run_experiment(spark: SparkSession, fraction: float = 0.2):
    """Run sentiment analysis and keyword extraction on a fraction of the dataset."""
    logger.info(f"Running with fraction {fraction:.2f}")
    metrics: Dict[str, Union[float, int]] = {"fraction": fraction}

    try:
        # Read and preprocess data
        start = time.time()
        df = spark.read.parquet(INPUT_PATH).sample(fraction)
        df = df.filter(col("cleaned_reviewText").isNotNull() & (trim(col("cleaned_reviewText")) != ""))
        # Add dummy row to ensure non-empty DataFrame
        dummy_row = spark.createDataFrame([("dummy text",)], ["cleaned_reviewText"])
        df = df.union(dummy_row).repartition(400).cache()
        metrics["total_records"] = df.count()
        metrics["read_time_seconds"] = time.time() - start

        # Keyword extraction using Spark ML
        keyword_start = time.time()
        tokenizer = Tokenizer(inputCol="cleaned_reviewText", outputCol="words")
        df = tokenizer.transform(df)
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        words_df = remover.transform(df)
        words_exploded = words_df.select(explode(col("filtered_words")).alias("word")).filter(col("word") != "")
        word_freq = words_exploded.groupBy("word").count().withColumnRenamed("count", "frequency").orderBy(col("frequency").desc())
        metrics["keyword_time_seconds"] = time.time() - keyword_start
        # Save keyword results to S3
        output_dir = f"{KEYWORD_OUTPUT_PATH}fraction_{fraction:.2f}/"
        word_freq.write.mode("overwrite").parquet(output_dir)
        metrics["keyword_save_time_seconds"] = time.time() - keyword_start

        # Save top 10 words as JSON
        top_words_df = word_freq.limit(10).select("word", "frequency").toPandas()
        top_words_json = top_words_df.to_dict(orient='records')
        with open(f"/tmp/top_words_fraction_{fraction:.2f}.json", "w") as f:
            json.dump(top_words_json, f, indent=4)
        upload_to_s3(f"/tmp/top_words_fraction_{fraction:.2f}.json", f"{KEYWORD_JSON_PATH}top_words_fraction_{fraction:.2f}.json")

        # Sentiment analysis using TextBlob
        sentiment_start = time.time()
        sentiment_df = df.select(
            col("cleaned_reviewText"),
            sentiment_udf(col("cleaned_reviewText")).alias("sentiment")
        ).select(
            col("cleaned_reviewText"),
            col("sentiment.polarity").alias("compound"),
            col("sentiment.category").alias("category")
        ).filter(col("cleaned_reviewText") != "dummy text")
        avg_sentiment = sentiment_df.selectExpr("avg(compound)").collect()[0][0] or 0.0
        category_counts = sentiment_df.groupBy("category").count().collect()
        category_counts = [(row["category"], row["count"]) for row in category_counts]
        metrics["sentiment_time_seconds"] = time.time() - sentiment_start
        metrics["avg_sentiment"] = avg_sentiment
        for cat, count in category_counts:
            metrics[f"{cat}_count"] = count
        # Save sentiment results to S3
        output_dir = f"{SENTIMENT_OUTPUT_PATH}fraction_{fraction:.2f}/"
        sentiment_df.write.mode("overwrite").parquet(output_dir)
        metrics["sentiment_save_time_seconds"] = time.time() - sentiment_start

        # Save sentiment categories as JSON
        category_counts_json = [{"category": cat, "count": count} for cat, count in category_counts]
        with open(f"/tmp/categories_fraction_{fraction:.2f}.json", "w") as f:
            json.dump(category_counts_json, f, indent=4)
        upload_to_s3(f"/tmp/categories_fraction_{fraction:.2f}.json", f"{SENTIMENT_JSON_PATH}categories_fraction_{fraction:.2f}.json")

        # Generate and save plots
        plot_top_words(top_words_df, fraction)
        plot_sentiment_categories(category_counts, fraction)

        # Calculate final metrics
        metrics["total_execution_time_seconds"] = time.time() - start
        metrics["throughput_records_per_second"] = metrics["total_records"] / metrics["total_execution_time_seconds"] if metrics["total_execution_time_seconds"] > 0 else 0
        metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if metrics["total_records"] > 0 else 0
        metrics["partition_count"] = df.select(spark_partition_id()).distinct().count()

        # Save metrics as JSON
        with open(f"/tmp/metrics_fraction_{fraction:.2f}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        upload_to_s3(f"/tmp/metrics_fraction_{fraction:.2f}.json", f"{METRICS_PATH}metrics_fraction_{fraction:.2f}.json")

        return metrics

    except Exception as e:
        logger.error(f"Error in run_experiment for fraction {fraction:.2f}: {e}")
        raise


def main():
    """Initialize Spark session and run the experiment for 20% of the dataset."""
    # Configure Spark session
    spark = SparkSession.builder \
        .appName("CombinedAnalysisFraction0.2") \
        .config("spark.sql.shuffle.partitions", 400) \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.hadoop.fs.s3a.fast.upload", "true") \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()
    
    try:
        run_experiment(spark, fraction=0.2)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

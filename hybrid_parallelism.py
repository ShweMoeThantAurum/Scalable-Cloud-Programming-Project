import os
os.environ['NLTK_DATA'] = '/usr/local/share/nltk_data'
import boto3
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, explode
import psutil
import pandas as pd
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# AWS and S3 configuration
s3_client = boto3.client('s3')
bucket_name = 'electronic-reviews-bucket'
input_prefix = 'processed_text_data_original/'
json_prefix = 'processed_text_data_json/'
output_prefix = 'hybrid_results/'

# Constants
CHUNK_SIZES = {
    "20%": 20000,
    "40%": 40000,
    "60%": 60000,
    "80%": 80000,
    "100%": 100000
}
EC2_TASKS = ['sentiment_analysis', 'report_generation', 'resource_monitoring']

def mapreduce_word_count(spark, bucket_name, input_prefix, chunk_size):
    start_time = time.time()
    try:
        df = spark.read.parquet(f"s3a://{bucket_name}/{input_prefix}part-*.parquet").limit(chunk_size)
        tokenizer = Tokenizer(inputCol="cleaned_reviewText", outputCol="words")
        df = tokenizer.transform(df)
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        words_df = remover.transform(df)
        word_freq = words_df.select(explode(col("filtered_words")).alias("word")).groupBy("word").count()
        top_words = word_freq.orderBy(col("count").desc()).limit(10).collect()
        execution_time = time.time() - start_time
        logger.info(f"MapReduce word count completed in {execution_time:.2f} seconds")
        return top_words, execution_time
    except Exception as e:
        logger.error(f"MapReduce error: {e}")
        raise

def sentiment_analysis_task(spark, bucket_name, json_prefix):
    start_time = time.time()
    try:
        df = spark.read.json(f"s3a://{bucket_name}/{json_prefix}part-*.json").limit(1000)
        pandas_df = df.select("cleaned_reviewText").toPandas()
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for _, row in pandas_df.iterrows():
            from textblob import TextBlob
            blob = TextBlob(row['cleaned_reviewText'])
            polarity = blob.sentiment.polarity
            category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral'
            sentiment_counts[category] += 1
        execution_time = time.time() - start_time
        logger.info(f"Sentiment analysis completed in {execution_time:.2f} seconds")
        return sentiment_counts, execution_time
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise

def report_generation_task():
    start_time = time.time()
    try:
        report = {"summary": "Top words and sentiment analysis completed", "timestamp": time.time()}
        output_key = f"{output_prefix}report_{int(time.time())}.json"
        s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=json.dumps(report))
        execution_time = time.time() - start_time
        logger.info(f"Report generation completed in {execution_time:.2f} seconds")
        return report, execution_time
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise

def resource_monitoring_task():
    start_time = time.time()
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        metrics = {"cpu_usage_percent": cpu_usage, "memory_usage_percent": memory_usage}
        output_key = f"{output_prefix}metrics_{int(time.time())}.json"
        s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=json.dumps(metrics))
        execution_time = time.time() - start_time
        logger.info(f"Resource monitoring completed in {execution_time:.2f} seconds")
        return metrics, execution_time
    except Exception as e:
        logger.error(f"Resource monitoring error: {e}")
        raise

def sequential_execution(spark, bucket_name, input_prefix, chunk_size):
    start_time = time.time()
    total_cpu = psutil.cpu_percent(interval=1)
    total_memory = psutil.virtual_memory().percent
    results = {}

    results['mapreduce'], mapreduce_time = mapreduce_word_count(spark, bucket_name, input_prefix, chunk_size)
    total_cpu += psutil.cpu_percent(interval=1)
    total_memory += psutil.virtual_memory().percent

    results['sentiment'], sentiment_time = sentiment_analysis_task(spark, bucket_name, json_prefix)
    total_cpu += psutil.cpu_percent(interval=1)
    total_memory += psutil.virtual_memory().percent

    results['report'], report_time = report_generation_task()
    total_cpu += psutil.cpu_percent(interval=1)
    total_memory += psutil.virtual_memory().percent

    results['metrics'], metrics_time = resource_monitoring_task()
    total_cpu += psutil.cpu_percent(interval=1)
    total_memory += psutil.virtual_memory().percent

    total_time = time.time() - start_time
    avg_cpu = total_cpu / 4
    avg_memory = total_memory / 4
    logger.info(f"Sequential execution completed in {total_time:.2f} seconds")
    return {
        "results": results,
        "total_time": total_time,
        "times": {
            "mapreduce": mapreduce_time,
            "sentiment": sentiment_time,
            "report": report_time,
            "metrics": metrics_time
        },
        "cpu_usage_percent": avg_cpu,
        "memory_usage_percent": avg_memory
    }

def parallel_execution(spark, bucket_name, input_prefix, chunk_size):
    start_time = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=len(EC2_TASKS) + 1) as executor:
        mapreduce_future = executor.submit(mapreduce_word_count, spark, bucket_name, input_prefix, chunk_size)
        sentiment_future = executor.submit(sentiment_analysis_task, spark, bucket_name, json_prefix)
        report_future = executor.submit(report_generation_task)
        metrics_future = executor.submit(resource_monitoring_task)

        results['mapreduce'], mapreduce_time = mapreduce_future.result()
        results['sentiment'], sentiment_time = sentiment_future.result()
        results['report'], report_time = report_future.result()
        results['metrics'], metrics_time = metrics_future.result()

    total_cpu = psutil.cpu_percent(interval=1)
    total_memory = psutil.virtual_memory().percent
    total_time = time.time() - start_time
    logger.info(f"Parallel execution completed in {total_time:.2f} seconds")
    return {
        "results": results,
        "total_time": total_time,
        "times": {
            "mapreduce": mapreduce_time,
            "sentiment": sentiment_time,
            "report": report_time,
            "metrics": metrics_time
        },
        "cpu_usage_percent": total_cpu,
        "memory_usage_percent": total_memory
    }

def plot_benchmarks(sequential_metrics, parallel_metrics, title_suffix, percent_label):
    tasks = ['mapreduce', 'sentiment', 'report', 'metrics']
    sequential_times = [sequential_metrics['times'][task] for task in tasks]
    parallel_times = [parallel_metrics['times'][task] for task in tasks]
    sequential_labels = [f"{t:.3f} s" for t in sequential_times]
    parallel_labels = [f"{t:.3f} s" for t in parallel_times]

    fig = go.Figure(data=[
        go.Bar(name='Sequential', x=tasks, y=sequential_times, marker_color='#1f77b4',
               text=sequential_labels, textposition='auto'),
        go.Bar(name='Parallel', x=tasks, y=parallel_times, marker_color='#d62728',
               text=parallel_labels, textposition='auto')
    ])
    fig.update_layout(
        title=f"Sequential vs Parallel Execution Times of {percent_label} of the Dataset",
        xaxis_title='Task Names',
        yaxis_title='Execution Time (s)',
        barmode='group',
        height=600,
        width=800,
        font=dict(family="Arial", size=14),
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        paper_bgcolor='white',
        legend_title='Execution Mode',
        xaxis=dict(tickangle=0, tickfont=dict(size=12))
    )
    path = f"/tmp/benchmarks_{title_suffix}.png"
    fig.write_image(path, format="png", scale=2)
    s3_client.upload_file(path, bucket_name, f"{output_prefix}benchmarks_{title_suffix}.png")
    logger.info(f"Saved benchmark plot to s3://{bucket_name}/{output_prefix}benchmarks_{title_suffix}.png")

def main():
    credentials = boto3.Session().get_credentials()
    access_key = credentials.access_key
    secret_key = credentials.secret_key

    spark = SparkSession.builder \
        .appName("HybridParallelism") \
        .config("spark.hadoop.fs.s3a.access.key", access_key) \
        .config("spark.hadoop.fs.s3a.secret.key", secret_key) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
        .config("spark.hadoop.fs.s3a.attempts.maximum", "10") \
        .getOrCreate()

    try:
        for percent_label, chunk_size in CHUNK_SIZES.items():
            logger.info(f"--- Benchmarking with {percent_label} of the dataset ({chunk_size} records) ---")

            logger.info("Starting sequential execution")
            sequential_metrics = sequential_execution(spark, bucket_name, input_prefix, chunk_size)

            logger.info("Starting parallel execution")
            parallel_metrics = parallel_execution(spark, bucket_name, input_prefix, chunk_size)

            speedup = sequential_metrics['total_time'] / parallel_metrics['total_time'] if parallel_metrics['total_time'] > 0 else 1
            logger.info(f"Speedup factor for {percent_label}: {speedup:.2f}x")

            metrics = {
                "sequential": sequential_metrics,
                "parallel": parallel_metrics,
                "speedup": speedup
            }
            output_key = f"{output_prefix}hybrid_metrics_{percent_label.replace('%','')}_{int(time.time())}.json"
            s3_client.put_object(Bucket=bucket_name, Key=output_key, Body=json.dumps(metrics))
            logger.info(f"Saved metrics to s3://{bucket_name}/{output_key}")

            plot_benchmarks(sequential_metrics, parallel_metrics, title_suffix=percent_label.replace("%", ""), percent_label=percent_label)

    except Exception as e:
        logger.error(f"Error in hybrid parallelism: {e}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main()

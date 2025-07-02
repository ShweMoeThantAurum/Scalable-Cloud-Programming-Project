import logging
import boto3
import json
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, spark_partition_id
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_PATH = "s3://electronics-reviews-bucket/processed_text_data/"
OUTPUT_PATH = "s3://electronics-reviews-bucket/keyword_results/"
BUCKET_NAME = "electronics-reviews-bucket"
OUTPUT_PREFIX = "keyword_results/"
METRICS_PATH = "results/metrics/"
PLOT_PATH = "results/plots/"
TOTAL_DATA_SIZE_GB = 3.33

def plot_metrics(metrics_list, output_path, bucket_name):
    """Generate and save performance plots using Plotly."""
    df = pd.DataFrame(metrics_list)
    
    # Create subplots for throughput and latency
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Throughput vs. Data Size", "Latency vs. Data Size"),
        vertical_spacing=0.15
    )
    
    # Add throughput plot
    fig.add_trace(
        go.Scatter(
            x=df['data_size_gb'],
            y=df['throughput_records_per_second'],
            mode='lines+markers',
            name='Throughput',
            line=dict(color='royalblue', width=2),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add latency plot
    fig.add_trace(
        go.Scatter(
            x=df['data_size_gb'],
            y=df['latency_per_record_ms'],
            mode='lines+markers',
            name='Latency',
            line=dict(color='crimson', width=2),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Update layout for aesthetics
    fig.update_layout(
        height=800,
        width=1000,
        showlegend=True,
        title="Performance Metrics Across Data Sizes",
        title_x=0.5,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(title_text="Data Size (GB)", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="Data Size (GB)", row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Throughput (records/s)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="Latency (ms/record)", row=2, col=1, gridcolor='lightgray')
    
    # Save as PNG for IEEE report
    local_plot_path = "/tmp/performance_plots.png"
    fig.write_image(local_plot_path, format="png", scale=2)  # High resolution
    
    # Save as HTML for demo video
    local_html_path = "/tmp/performance_plots.html"
    fig.write_html(local_html_path)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_plot_path, bucket_name, f"{output_path}performance_plots.png")
    s3_client.upload_file(local_html_path, bucket_name, f"{output_path}performance_plots.html")
    logger.info(f"Performance plots saved to s3://{bucket_name}/{output_path}performance_plots.png")
    logger.info(f"Interactive HTML plot saved to s3://{bucket_name}/{output_path}performance_plots.html")

def plot_top_words(word_freq, data_size_gb, output_path, bucket_name):
    """Generate and save bar chart for top 10 words using Plotly."""
    # Convert top 10 words to pandas for plotting
    top_words_df = word_freq.limit(10).select("word", "frequency").toPandas()
    
    # Create bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=top_words_df['word'],
                y=top_words_df['frequency'],
                marker_color='teal',
                text=top_words_df['frequency'],
                textposition='auto'
            )
        ]
    )
    
    # Update layout for aesthetics
    fig.update_layout(
        height=600,
        width=800,
        title=f"Top 10 Words for Data Size {data_size_gb:.2f} GB",
        title_x=0.5,
        font=dict(size=14),
        xaxis_title="Word",
        yaxis_title="Frequency",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45, gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray')
    )
    
    # Save as PNG
    local_plot_path = f"/tmp/top_words_data_size={data_size_gb:.2f}GB.png"
    fig.write_image(local_plot_path, format="png", scale=2)
    
    # Save as HTML
    local_html_path = f"/tmp/top_words_data_size={data_size_gb:.2f}GB.html"
    fig.write_html(local_html_path)
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_plot_path, bucket_name, f"{output_path}top_words_data_size={data_size_gb:.2f}GB.png")
    s3_client.upload_file(local_html_path, bucket_name, f"{output_path}top_words_data_size={data_size_gb:.2f}GB.html")
    logger.info(f"Top 10 words plot saved to s3://{bucket_name}/{output_path}top_words_data_size={data_size_gb:.2f}GB.png")
    logger.info(f"Interactive top 10 words HTML saved to s3://{bucket_name}/{output_path}top_words_data_size={data_size_gb:.2f}GB.html")

def run_experiment(spark, fraction, input_path, output_path, bucket_name, output_prefix, total_size_gb=3.33):
    """Run word count experiment with specified data fraction."""
    logger.info(f"Running experiment with data fraction {fraction:.2f} (~{fraction*total_size_gb:.2f} GB)")
    
    metrics = {"data_size_gb": fraction * total_size_gb}
    start_time = time.time()
    
    # Step 1: Read parquet files, sample, and cache
    df = spark.read.parquet(input_path).sample(fraction=fraction).cache()
    metrics["total_records"] = df.count()
    metrics["read_time_seconds"] = time.time() - start_time
    logger.info(f"Read {metrics['total_records']} records in {metrics['read_time_seconds']:.2f} seconds")
    
    # Step 2: Tokenize
    tokenize_start = time.time()
    tokenizer = Tokenizer(inputCol="cleaned_reviewText", outputCol="words")
    df = tokenizer.transform(df)
    metrics["tokenize_time_seconds"] = time.time() - tokenize_start
    
    # Step 3: Remove stop words
    stopwords_start = time.time()
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df = remover.transform(df)
    metrics["stopwords_time_seconds"] = time.time() - stopwords_start
    
    # Step 4: Calculate word frequency
    wordfreq_start = time.time()
    df_words = df.select(explode(col("filtered_words")).alias("word")) \
                 .filter(col("word") != "")
    word_freq = df_words.groupBy("word").count() \
                       .withColumnRenamed("count", "frequency") \
                       .orderBy(col("frequency").desc())
    metrics["wordfreq_time_seconds"] = time.time() - wordfreq_start
    
    # Step 5: Save all word frequencies and log top 5
    save_start = time.time()
    output_dir = f"{output_path}data_size={fraction*total_size_gb:.2f}GB/"
    word_freq.write.mode("overwrite").parquet(output_dir)
    metrics["save_time_seconds"] = time.time() - save_start
    
    # Log top 5 words for reference
    top_words = word_freq.limit(5).select("word", "frequency")
    top_words.show(5, truncate=False)
    
    # Step 6: Generate top 10 words visualization
    plot_top_words(word_freq, fraction * total_size_gb, PLOT_PATH, BUCKET_NAME)
    
    # Step 7: Calculate throughput, latency, and partition count
    metrics["total_execution_time_seconds"] = time.time() - start_time
    metrics["throughput_records_per_second"] = metrics["total_records"] / metrics["total_execution_time_seconds"] if metrics["total_execution_time_seconds"] > 0 else 0
    metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if metrics["total_records"] > 0 else 0
    metrics["partition_count"] = df.select(spark_partition_id()).distinct().count()
    
    return metrics

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("KeywordExtraction").getOrCreate()
    
    try:
        # Define data fractions for subsets (20%, 40%, 60%, 80%, 100%)
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]  # Correspond to 0.67 GB, 1.33 GB, 2.00 GB, 2.66 GB, 3.33 GB
        metrics_list = []
        for fraction in fractions:
            metrics = run_experiment(spark, fraction, INPUT_PATH, OUTPUT_PATH, BUCKET_NAME, OUTPUT_PREFIX, TOTAL_DATA_SIZE_GB)
            metrics_list.append(metrics)
        
        # Save metrics as JSON
        metrics_json_path = "/tmp/keyword_metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_list, f, indent=4)
        s3_client = boto3.client('s3')
        s3_client.upload_file(metrics_json_path, BUCKET_NAME, f"{METRICS_PATH}keyword_metrics.json")
        logger.info(f"Metrics JSON saved to s3://{BUCKET_NAME}/{METRICS_PATH}keyword_metrics.json")
        
        # Generate and save performance plots
        plot_metrics(metrics_list, PLOT_PATH, BUCKET_NAME)
        
        # Save Spark configuration for report
        spark_config = {
            "spark.app.name": spark.conf.get("spark.app.name"),
            "spark.master": spark.conf.get("spark.master"),
            "spark.executor.cores": spark.conf.get("spark.executor.cores", "default"),
            "spark.executor.memory": spark.conf.get("spark.executor.memory", "default"),
            "spark.driver.memory": spark.conf.get("spark.driver.memory", "default"),
            "spark.sql.shuffle.partitions": int(spark.conf.get("spark.sql.shuffle.partitions", "200"))
        }
        config_path = "/tmp/spark_config.json"
        with open(config_path, "w") as f:
            json.dump(spark_config, f, indent=4)
        s3_client.upload_file(config_path, BUCKET_NAME, f"{METRICS_PATH}spark_config.json")
        logger.info(f"Spark config saved to s3://{BUCKET_NAME}/{METRICS_PATH}spark_config.json")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
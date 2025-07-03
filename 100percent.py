<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Guide: Processing Electronics Dataset</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <style>
        .code-block {
            background-color: #f6f8fa;
            border-radius: 8px;
            overflow-x: auto;
        }

        .code-block pre {
            padding: 1.5rem;
            margin: 0;
        }

        .section-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .section-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>

<body class="bg-gradient-to-br from-gray-100 to-gray-200 font-sans text-gray-800 antialiased min-h-screen">
    <div class="container mx-auto px-4 py-12 max-w-5xl">
        <header class="text-center mb-12">
            <h1 class="text-5xl font-extrabold text-gray-900 mb-4 drop-shadow-md">Guide: Processing Electronics Dataset
            </h1>
            <p class="text-xl text-gray-700">Scalable Cloud-Based Sentiment Analysis for Amazon Electronics Reviews</p>
        </header>

        <main class="space-y-8">
            <section class="bg-white rounded-xl shadow-lg p-8 section-card" aria-labelledby="phase1-data-prep">
                <h2 id="phase1-data-prep"
                    class="text-3xl font-semibold text-gray-800 mb-6 border-b-2 border-indigo-500 pb-2">Phase 1: Data
                    Preparation</h2>
                <p class="text-lg text-gray-600 leading-relaxed">Upload the original dataset, Electronics.json (around
                    11 GB) to S3.</p>
            </section>

            <section class="bg-white rounded-xl shadow-lg p-8 section-card" aria-labelledby="phase1-emr">
                <h2 id="phase1-emr" class="text-3xl font-semibold text-gray-800 mb-6 border-b-2 border-indigo-500 pb-2">
                    Phase 1: AWS EMR Setup and Processing</h2>
                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">Create EMR Cluster</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Create a cluster on EMR with 2 cores. Use the
                    following bootstrap script:</p>
                <div class="code-block">
                    <pre><code class="language-bash">
#!/bin/bash
                    
# Exit on any error
set -e
                    
# Log file for debugging
LOG_FILE="/tmp/bootstrap.log"
echo "Starting bootstrap script..." | tee -a $LOG_FILE
                    
# Update system packages
echo "Updating system packages..." | tee -a $LOG_FILE
sudo yum update -y >> $LOG_FILE 2>&1
                    
# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
NLTK_DATA_PATH="/usr/local/lib/python${PYTHON_VERSION}/dist-packages/nltk_data"
echo "Detected Python version: $PYTHON_VERSION" | tee -a $LOG_FILE
echo "NLTK data path: $NLTK_DATA_PATH" | tee -a $LOG_FILE
                    
# Install Python dependencies
echo "Installing Python dependencies..." | tee -a $LOG_FILE
sudo pip3 install boto3 pandas plotly kaleido textblob numpy nltk >> $LOG_FILE 2>&1
                    
# Install NLTK data for TextBlob
echo "Installing NLTK data..." | tee -a $LOG_FILE
sudo mkdir -p $NLTK_DATA_PATH
sudo python3 -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA_PATH');
nltk.download('averaged_perceptron_tagger', download_dir='$NLTK_DATA_PATH'); nltk.download('brown',
download_dir='$NLTK_DATA_PATH')" >> $LOG_FILE 2>&1
                    
# Install Google Chrome for Kaleido (Plotly image rendering)
echo "Installing Google Chrome..." | tee -a $LOG_FILE
sudo wget https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm >> $LOG_FILE 2>&1
sudo yum install -y ./google-chrome-stable_current_x86_64.rpm >> $LOG_FILE 2>&1
sudo rm -f google-chrome-stable_current_x86_64.rpm >> $LOG_FILE 2>&1
                    
# Verify installations
echo "Verifying Python package installations..." | tee -a $LOG_FILE
pip3 show boto3 pandas plotly kaleido textblob numpy nltk >> $LOG_FILE 2>&1
                    
# Check NLTK data
echo "Verifying NLTK data..." | tee -a $LOG_FILE
ls -l $NLTK_DATA_PATH >> $LOG_FILE 2>&1
                    
echo "Bootstrap script completed successfully." | tee -a $LOG_FILE
exit 0
                    </code></pre>
                </div>
                <p class="mt-6 text-lg text-gray-600 leading-relaxed">Write the script on your local machine via Visual
                    Studio Code, save it, upload to S3, and use it as the bootstrap file when creating the cluster.</p>

                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">EMR Processing Script</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Run the following PySpark script (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/data_ingestion.py">data_ingestion.py</a>)
                    on the EMR cluster:</p>
                <div class="code-block">
                    <pre><code class="language-python">
import logging
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, spark_partition_id
from pyspark.sql.types import StructType, StructField, StringType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INPUT_PATH = "s3://electronics-reviews-bucket/Electronics.json"
OUTPUT_PATH = "s3://electronics-reviews-bucket/processed_text_data_original/"
BUCKET_NAME = "electronics-reviews-bucket"
OUTPUT_PREFIX = "processed_text_data_original/"

def main():
    spark = SparkSession.builder.appName("ElectronicsTextDataIngestion").getOrCreate()
    schema = StructType([StructField("reviewText", StringType(), True)])
    try:
        df_text = spark.read.option("multiLine", False).schema(schema).json(INPUT_PATH) \
            .filter(col("reviewText").isNotNull()) \
            .select(lower(col("reviewText")).alias("cleaned_reviewText"))
        df_text.write.mode("overwrite").parquet(OUTPUT_PATH)
        record_count = df_text.count()
        partition_count = df_text.select(spark_partition_id()).distinct().count()
        logger.info(f"Total records ingested: {record_count}")
        logger.info(f"Number of partitions: {partition_count}")
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
                    </code></pre>
                </div>
            </section>

            <section class="bg-white rounded-xl shadow-lg p-8 section-card" aria-labelledby="phase2">
                <h2 id="phase2" class="text-3xl font-semibold text-gray-800 mb-6 border-b-2 border-indigo-500 pb-2">
                    Phase 2: MapReduce Implementation</h2>
                <p class="text-lg text-gray-600 mb-6 leading-relaxed">Add the following files to the project:</p>
                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">20percent.py</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Run this script (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/20percent.py">20percent.py</a>)
                    for 20% data processing:</p>
                <div class="code-block">
                    <pre><code class="language-python">
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
                        category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral' return (polarity, category)
                            except Exception as e: logger.error(f"Error in sentiment analysis: {e}") return (0.0, "neutral" ) # Register UDF for
                            sentiment analysis sentiment_udf=udf( map_sentiment_udf, StructType([ StructField("polarity", FloatType(), False),
                            StructField("category", StringType(), False) ]) ) def upload_to_s3(local_path: str, s3_key: str, max_retries:
                            int=3): """Upload a file to S3 with exponential backoff retries.""" s3_client=boto3.client('s3') for attempt in
                            range(max_retries): try: s3_client.upload_file(local_path, BUCKET_NAME, s3_key) logger.info(f"Uploaded to
                            s3://{BUCKET_NAME}/{s3_key}") return except Exception as e: logger.warning(f"S3 upload attempt {attempt + 1} failed:
                            {e}") if attempt==max_retries - 1: logger.error(f"Failed to upload to s3://{BUCKET_NAME}/{s3_key} after
                            {max_retries} attempts") raise time.sleep(2 ** attempt) def plot_top_words(top_words_df: pd.DataFrame, fraction:
                            float): """Generate and save a bar chart of the top 10 words with their frequencies.""" colors=['#1f77b4', '#2ca02c'
                            , '#d62728' , '#9467bd' , '#8c564b' , '#e377c2' , '#7f7f7f' , '#bcbd22' , '#17becf' , '#ff7f0e' ]
                            fig=go.Figure([go.Bar( x=top_words_df['word'], y=top_words_df['frequency'], text=top_words_df['frequency'],
                            textposition='auto' , marker_color=colors[:len(top_words_df)], marker_line_color='black' , marker_line_width=1.5,
                            opacity=0.85 )]) # Set dynamic title based on fraction title="Top 10 Words for 20% of the Dataset" if fraction==0.2
                            else f"Top 10 Words for {int(fraction*100)}% of the Dataset" if fraction==1.0:
                            title="Top 10 Words for the Whole Dataset" fig.update_layout( title=title, xaxis_title="Word" ,
                            yaxis_title="Frequency" , title_x=0.5, height=600, width=800, font=dict(family="Arial, sans-serif" , size=16,
                            color="black" ), plot_bgcolor='rgba(240, 240, 240, 0.95)' , paper_bgcolor='white' , xaxis=dict(tickangle=45,
                            gridcolor='lightgray' , title_font=dict(size=18), tickfont=dict(size=14)), yaxis=dict(gridcolor='lightgray' ,
                            title_font=dict(size=18), tickfont=dict(size=14)), showlegend=False, margin=dict(l=50, r=50, t=100, b=100) ) # Save
                            plot to temporary file and upload to S3 path=f"/tmp/top_words_fraction_{fraction:.2f}.png" fig.write_image(path,
                            format="png" , scale=2) upload_to_s3(path, f"{KEYWORD_PLOT_PATH}top_words_fraction_{fraction:.2f}.png") def
                            plot_sentiment_categories(category_counts: List[Tuple[str, int]], fraction:
                            float): """Generate and save a pie chart of sentiment category counts.""" df=pd.DataFrame(category_counts,
                            columns=['category', 'count' ]) colors=['#1f77b4', '#2ca02c' , '#d62728' ] # Colors for positive, neutral, negative
                            fig=go.Figure([go.Pie( labels=df['category'], values=df['count'], textinfo='label+percent' , textposition='inside' ,
                            marker=dict(colors=colors[:len(df)], line=dict(color='black' , width=1.5)), opacity=0.85 )])
                            title="Sentiment Analysis for 20% of the Dataset" if fraction==0.2 else f"Sentiment Analysis for
                            {int(fraction*100)}% of the Dataset" if fraction==1.0: title="Sentiment Analysis for the Whole Dataset"
                            fig.update_layout( title=title, title_x=0.5, height=600, width=800, font=dict(family="Arial, sans-serif" , size=16,
                            color="black" ), plot_bgcolor='rgba(240, 240, 240, 0.95)' , paper_bgcolor='white' , showlegend=True,
                            legend=dict(title="Sentiment Categories" , font=dict(size=14)), margin=dict(l=50, r=50, t=100, b=100) )
                            path=f"/tmp/categories_fraction_{fraction:.2f}.png" fig.write_image(path, format="png" , scale=2) upload_to_s3(path,
                            f"{SENTIMENT_PLOT_PATH}categories_fraction_{fraction:.2f}.png") def run_experiment(spark: SparkSession, fraction:
                            float=0.2): """Run sentiment analysis and keyword extraction on a fraction of the dataset.""" logger.info(f"Running
                            with fraction {fraction:.2f}") metrics: Dict[str, Union[float, int]]={"fraction": fraction} try: # Read and
                            preprocess data start=time.time() df=spark.read.parquet(INPUT_PATH).sample(fraction)
                            df=df.filter(col("cleaned_reviewText").isNotNull() & (trim(col("cleaned_reviewText")) !="" )) # Add dummy row to
                            ensure non-empty DataFrame dummy_row=spark.createDataFrame([("dummy text",)], ["cleaned_reviewText"])
                            df=df.union(dummy_row).repartition(400).cache() metrics["total_records"]=df.count()
                            metrics["read_time_seconds"]=time.time() - start # Keyword extraction using Spark ML keyword_start=time.time()
                            tokenizer=Tokenizer(inputCol="cleaned_reviewText" , outputCol="words" ) df=tokenizer.transform(df)
                            remover=StopWordsRemover(inputCol="words" , outputCol="filtered_words" ) words_df=remover.transform(df)
                            words_exploded=words_df.select(explode(col("filtered_words")).alias("word")).filter(col("word") !="" )
                            word_freq=words_exploded.groupBy("word").count().withColumnRenamed("count", "frequency"
                            ).orderBy(col("frequency").desc()) metrics["keyword_time_seconds"]=time.time() - keyword_start # Save keyword
                            results to S3 output_dir=f"{KEYWORD_OUTPUT_PATH}fraction_{fraction:.2f}/"
                            word_freq.write.mode("overwrite").parquet(output_dir) metrics["keyword_save_time_seconds"]=time.time() -
                            keyword_start # Save top 10 words as JSON top_words_df=word_freq.limit(10).select("word", "frequency" ).toPandas()
                            top_words_json=top_words_df.to_dict(orient='records' ) with open(f"/tmp/top_words_fraction_{fraction:.2f}.json", "w"
                            ) as f: json.dump(top_words_json, f, indent=4) upload_to_s3(f"/tmp/top_words_fraction_{fraction:.2f}.json",
                            f"{KEYWORD_JSON_PATH}top_words_fraction_{fraction:.2f}.json") # Sentiment analysis using TextBlob
                            sentiment_start=time.time() sentiment_df=df.select( col("cleaned_reviewText"),
                            sentiment_udf(col("cleaned_reviewText")).alias("sentiment") ).select( col("cleaned_reviewText"),
                            col("sentiment.polarity").alias("compound"), col("sentiment.category").alias("category")
                            ).filter(col("cleaned_reviewText") !="dummy text" )
                            avg_sentiment=sentiment_df.selectExpr("avg(compound)").collect()[0][0] or 0.0
                            category_counts=sentiment_df.groupBy("category").count().collect() category_counts=[(row["category"], row["count"])
                            for row in category_counts] metrics["sentiment_time_seconds"]=time.time() - sentiment_start
                            metrics["avg_sentiment"]=avg_sentiment for cat, count in category_counts: metrics[f"{cat}_count"]=count # Save
                            sentiment results to S3 output_dir=f"{SENTIMENT_OUTPUT_PATH}fraction_{fraction:.2f}/"
                            sentiment_df.write.mode("overwrite").parquet(output_dir) metrics["sentiment_save_time_seconds"]=time.time() -
                            sentiment_start # Save sentiment categories as JSON category_counts_json=[{"category": cat, "count" : count} for
                            cat, count in category_counts] with open(f"/tmp/categories_fraction_{fraction:.2f}.json", "w" ) as f:
                            json.dump(category_counts_json, f, indent=4) upload_to_s3(f"/tmp/categories_fraction_{fraction:.2f}.json",
                            f"{SENTIMENT_JSON_PATH}categories_fraction_{fraction:.2f}.json") # Generate and save plots
                            plot_top_words(top_words_df, fraction) plot_sentiment_categories(category_counts, fraction) # Calculate final
                            metrics metrics["total_execution_time_seconds"]=time.time() - start
                            metrics["throughput_records_per_second"]=metrics["total_records"] / metrics["total_execution_time_seconds"] if
                            metrics["total_execution_time_seconds"]> 0 else 0
                            metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if
                            metrics["total_records"] > 0 else 0
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
                        
                        
                            if __name__ == "__main__":                  main()
                    </code></pre>
                </div>

                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">40percent.py</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Run this script (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/40percent.py">40percent.py</a>)
                    for 40% data processing:</p>
                <div class="code-block">
                    <pre><code class="language-python">
                        """Process 40% of a dataset for sentiment analysis and keyword extraction using Spark."""
                        
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
                        category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral' return (polarity, category)
                            except Exception as e: logger.error(f"Error in sentiment analysis: {e}") return (0.0, "neutral" ) # Register UDF for
                            sentiment analysis sentiment_udf=udf( map_sentiment_udf, StructType([ StructField("polarity", FloatType(), False),
                            StructField("category", StringType(), False) ]) ) def upload_to_s3(local_path: str, s3_key: str, max_retries:
                            int=3): """Upload a file to S3 with exponential backoff retries.""" s3_client=boto3.client('s3') for attempt in
                            range(max_retries): try: s3_client.upload_file(local_path, BUCKET_NAME, s3_key) logger.info(f"Uploaded to
                            s3://{BUCKET_NAME}/{s3_key}") return except Exception as e: logger.warning(f"S3 upload attempt {attempt + 1} failed:
                            {e}") if attempt==max_retries - 1: logger.error(f"Failed to upload to s3://{BUCKET_NAME}/{s3_key} after
                            {max_retries} attempts") raise time.sleep(2 ** attempt) def plot_top_words(top_words_df: pd.DataFrame, fraction:
                            float): """Generate and save a bar chart of the top 10 words with their frequencies.""" colors=['#1f77b4', '#2ca02c'
                            , '#d62728' , '#9467bd' , '#8c564b' , '#e377c2' , '#7f7f7f' , '#bcbd22' , '#17becf' , '#ff7f0e' ]
                            fig=go.Figure([go.Bar( x=top_words_df['word'], y=top_words_df['frequency'], text=top_words_df['frequency'],
                            textposition='auto' , marker_color=colors[:len(top_words_df)], marker_line_color='black' , marker_line_width=1.5,
                            opacity=0.85 )]) # Set dynamic title based on fraction title="Top 10 Words for 20% of the Dataset" if fraction==0.2
                            else f"Top 10 Words for {int(fraction*100)}% of the Dataset" if fraction==1.0:
                            title="Top 10 Words for the Whole Dataset" fig.update_layout( title=title, xaxis_title="Word" ,
                            yaxis_title="Frequency" , title_x=0.5, height=600, width=800, font=dict(family="Arial, sans-serif" , size=16,
                            color="black" ), plot_bgcolor='rgba(240, 240, 240, 0.95)' , paper_bgcolor='white' , xaxis=dict(tickangle=45,
                            gridcolor='lightgray' , title_font=dict(size=18), tickfont=dict(size=14)), yaxis=dict(gridcolor='lightgray' ,
                            title_font=dict(size=18), tickfont=dict(size=14)), showlegend=False, margin=dict(l=50, r=50, t=100, b=100) ) # Save
                            plot to temporary file and upload to S3 path=f"/tmp/top_words_fraction_{fraction:.2f}.png" fig.write_image(path,
                            format="png" , scale=2) upload_to_s3(path, f"{KEYWORD_PLOT_PATH}top_words_fraction_{fraction:.2f}.png") def
                            plot_sentiment_categories(category_counts: List[Tuple[str, int]], fraction:
                            float): """Generate and save a pie chart of sentiment category counts.""" df=pd.DataFrame(category_counts,
                            columns=['category', 'count' ]) colors=['#1f77b4', '#2ca02c' , '#d62728' ] # Colors for positive, neutral, negative
                            fig=go.Figure([go.Pie( labels=df['category'], values=df['count'], textinfo='label+percent' , textposition='inside' ,
                            marker=dict(colors=colors[:len(df)], line=dict(color='black' , width=1.5)), opacity=0.85 )])
                            title="Sentiment Analysis for 20% of the Dataset" if fraction==0.2 else f"Sentiment Analysis for
                            {int(fraction*100)}% of the Dataset" if fraction==1.0: title="Sentiment Analysis for the Whole Dataset"
                            fig.update_layout( title=title, title_x=0.5, height=600, width=800, font=dict(family="Arial, sans-serif" , size=16,
                            color="black" ), plot_bgcolor='rgba(240, 240, 240, 0.95)' , paper_bgcolor='white' , showlegend=True,
                            legend=dict(title="Sentiment Categories" , font=dict(size=14)), margin=dict(l=50, r=50, t=100, b=100) )
                            path=f"/tmp/categories_fraction_{fraction:.2f}.png" fig.write_image(path, format="png" , scale=2) upload_to_s3(path,
                            f"{SENTIMENT_PLOT_PATH}categories_fraction_{fraction:.2f}.png") def run_experiment(spark: SparkSession, fraction:
                            float=0.4): """Run sentiment analysis and keyword extraction on a fraction of the dataset.""" logger.info(f"Running
                            with fraction {fraction:.2f}") metrics: Dict[str, Union[float, int]]={"fraction": fraction} try: # Read and
                            preprocess data start=time.time() df=spark.read.parquet(INPUT_PATH).sample(fraction)
                            df=df.filter(col("cleaned_reviewText").isNotNull() & (trim(col("cleaned_reviewText")) !="" )) # Add dummy row to
                            ensure non-empty DataFrame dummy_row=spark.createDataFrame([("dummy text",)], ["cleaned_reviewText"])
                            df=df.union(dummy_row).repartition(400).cache() metrics["total_records"]=df.count()
                            metrics["read_time_seconds"]=time.time() - start # Keyword extraction using Spark ML keyword_start=time.time()
                            tokenizer=Tokenizer(inputCol="cleaned_reviewText" , outputCol="words" ) df=tokenizer.transform(df)
                            remover=StopWordsRemover(inputCol="words" , outputCol="filtered_words" ) words_df=remover.transform(df)
                            words_exploded=words_df.select(explode(col("filtered_words")).alias("word")).filter(col("word") !="" )
                            word_freq=words_exploded.groupBy("word").count().withColumnRenamed("count", "frequency"
                            ).orderBy(col("frequency").desc()) metrics["keyword_time_seconds"]=time.time() - keyword_start # Save keyword
                            results to S3 output_dir=f"{KEYWORD_OUTPUT_PATH}fraction_{fraction:.2f}/"
                            word_freq.write.mode("overwrite").parquet(output_dir) metrics["keyword_save_time_seconds"]=time.time() -
                            keyword_start # Save top 10 words as JSON top_words_df=word_freq.limit(10).select("word", "frequency" ).toPandas()
                            top_words_json=top_words_df.to_dict(orient='records' ) with open(f"/tmp/top_words_fraction_{fraction:.2f}.json", "w"
                            ) as f: json.dump(top_words_json, f, indent=4) upload_to_s3(f"/tmp/top_words_fraction_{fraction:.2f}.json",
                            f"{KEYWORD_JSON_PATH}top_words_fraction_{fraction:.2f}.json") # Sentiment analysis using TextBlob
                            sentiment_start=time.time() sentiment_df=df.select( col("cleaned_reviewText"),
                            sentiment_udf(col("cleaned_reviewText")).alias("sentiment") ).select( col("cleaned_reviewText"),
                            col("sentiment.polarity").alias("compound"), col("sentiment.category").alias("category")
                            ).filter(col("cleaned_reviewText") !="dummy text" )
                            avg_sentiment=sentiment_df.selectExpr("avg(compound)").collect()[0][0] or 0.0
                            category_counts=sentiment_df.groupBy("category").count().collect() category_counts=[(row["category"], row["count"])
                            for row in category_counts] metrics["sentiment_time_seconds"]=time.time() - sentiment_start
                            metrics["avg_sentiment"]=avg_sentiment for cat, count in category_counts: metrics[f"{cat}_count"]=count # Save
                            sentiment results to S3 output_dir=f"{SENTIMENT_OUTPUT_PATH}fraction_{fraction:.2f}/"
                            sentiment_df.write.mode("overwrite").parquet(output_dir) metrics["sentiment_save_time_seconds"]=time.time() -
                            sentiment_start # Save sentiment categories as JSON category_counts_json=[{"category": cat, "count" : count} for
                            cat, count in category_counts] with open(f"/tmp/categories_fraction_{fraction:.2f}.json", "w" ) as f:
                            json.dump(category_counts_json, f, indent=4) upload_to_s3(f"/tmp/categories_fraction_{fraction:.2f}.json",
                            f"{SENTIMENT_JSON_PATH}categories_fraction_{fraction:.2f}.json") # Generate and save plots
                            plot_top_words(top_words_df, fraction) plot_sentiment_categories(category_counts, fraction) # Calculate final
                            metrics metrics["total_execution_time_seconds"]=time.time() - start
                            metrics["throughput_records_per_second"]=metrics["total_records"] / metrics["total_execution_time_seconds"] if
                            metrics["total_execution_time_seconds"]> 0 else 0
                            metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if
                            metrics["total_records"] > 0 else 0
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
                            """Initialize Spark session and run the experiment for 40% of the dataset."""
                            # Configure Spark session
                            spark = SparkSession.builder \
                            .appName("CombinedAnalysisFraction0.4") \
                            .config("spark.sql.shuffle.partitions", 400) \
                            .config("spark.executor.cores", "4") \
                            .config("spark.executor.memory", "8g") \
                            .config("spark.driver.memory", "8g") \
                            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
                            .config("spark.network.timeout", "600s") \
                            .config("spark.executor.heartbeatInterval", "60s") \
                            .getOrCreate()
                        
                            try:
                            run_experiment(spark, fraction=0.4)
                            except Exception as e:
                            logger.error(f"Error during processing: {e}")
                            raise
                            finally:
                            spark.stop()
                        
                        
                            if __name__ == "__main__":                  main()
                    </code></pre>
                </div>

                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">60percent.py</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Run this script (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/60percent.py">60percent.py</a>)
                    for 60% data processing:</p>
                <div class="code-block">
                    <pre><code class="language-python">
                        """Process 60% of a dataset for sentiment analysis and keyword extraction using Spark."""
                        
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
                        
                        
                        def map_sentiment_udf(text: Union[str, None]) -> Tuple[float, str]:
                        """Perform sentiment analysis on a given text using TextBlob."""
                        if not isinstance(text, str) or not text.strip():
                        logger.warning(f"Invalid text input: {text}")
                        return (0.0, "neutral")
                        try:
                        blob = TextBlob(text)
                        polarity = float(blob.sentiment.polarity)
                        category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral' return (polarity, category)
                            except Exception as e: logger.error(f"Error in sentiment analysis: {e}") return (0.0, "neutral" ) # Register UDF for
                            sentiment analysis sentiment_udf=udf( map_sentiment_udf, StructType([ StructField("polarity", FloatType(), False),
                            StructField("category", StringType(), False) ]) ) def upload_to_s3(local_path: str, s3_key: str, max_retries: int=3)
                            -> None:
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
                        
                        
                            def plot_top_words(top_words_df: pd.DataFrame, fraction: float) -> None:
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
                            title = "Top 10 Words for 20% of the Dataset" if fraction == 0.2 else f"Top 10 Words for {int(fraction*100)}% of the
                            Dataset"
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
                            colors = ['#1f77b4', '#2ca02c', '#d62728'] # Colors for positive, neutral, negative
                            fig = go.Figure([go.Pie(
                            labels=df['category'],
                            values=df['count'],
                            textinfo='label+percent',
                            textposition='inside',
                            marker=dict(colors=colors[:len(df)], line=dict(color='black', width=1.5)),
                            opacity=0.85
                            )])
                            title = "Sentiment Analysis for 20% of the Dataset" if fraction == 0.2 else f"Sentiment Analysis for
                            {int(fraction*100)}% of the Dataset"
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
                        
                        
                            def run_experiment(spark: SparkSession, fraction: float = 0.6) -> Dict[str, Union[float, int]]:
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
                            word_freq = words_exploded.groupBy("word").count().withColumnRenamed("count",
                            "frequency").orderBy(col("frequency").desc())
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
                            upload_to_s3(f"/tmp/top_words_fraction_{fraction:.2f}.json",
                            f"{KEYWORD_JSON_PATH}top_words_fraction_{fraction:.2f}.json")
                        
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
                            upload_to_s3(f"/tmp/categories_fraction_{fraction:.2f}.json",
                            f"{SENTIMENT_JSON_PATH}categories_fraction_{fraction:.2f}.json")
                        
                            # Generate and save plots
                            plot_top_words(top_words_df, fraction)
                            plot_sentiment_categories(category_counts, fraction)
                        
                            # Calculate final metrics
                            metrics["total_execution_time_seconds"] = time.time() - start
                            metrics["throughput_records_per_second"] = metrics["total_records"] / metrics["total_execution_time_seconds"] if
                            metrics["total_execution_time_seconds"] > 0 else 0
                            metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if
                            metrics["total_records"] > 0 else 0
                            metrics["partition_count"] = df.select(spark_partition_id()).distinct().count()
                        
                            # Save metrics as JSON
                            with open(f"/tmp/metrics_fraction_{fraction:.2f}.json", "w") as f:
                            json.dump(metrics, f, indent=4)
                            upload_to_s3(f"/tmp/metrics_fraction_{fraction:.2f}.json", f"{METRICS_PATH}metrics_fraction_{fraction:.2f}.json")
                        
                            return metrics
                        
                            except Exception as e:
                            logger.error(f"Error in run_experiment for fraction {fraction:.2f}: {e}")
                            raise
                        
                        
                            def main() -> None:
                            """Initialize Spark session and run the experiment for 60% of the dataset."""
                            # Configure Spark session
                            spark = SparkSession.builder \
                            .appName("CombinedAnalysisFraction0.6") \
                            .config("spark.sql.shuffle.partitions", 400) \
                            .config("spark.executor.cores", "4") \
                            .config("spark.executor.memory", "8g") \
                            .config("spark.driver.memory", "8g") \
                            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
                            .config("spark.network.timeout", "600s") \
                            .config("spark.executor.heartbeatInterval", "60s") \
                            .getOrCreate()
                        
                            try:
                            run_experiment(spark, fraction=0.6)
                            except Exception as e:
                            logger.error(f"Error during processing: {e}")
                            raise
                            finally:
                            spark.stop()
                        
                        
                            if __name__ == "__main__":                  main()
                    </code></pre>
                </div>

                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">80percent.py</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Run this script (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/80percent.py">80percent.py</a>)
                    for 80% data processing:</p>
                <div class="code-block">
                    <pre><code class="language-python">
                        """Process 80% of a dataset for sentiment analysis and keyword extraction using Spark."""
                        
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
                        category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral' return (polarity, category)
                            except Exception as e: logger.error(f"Error in sentiment analysis: {e}") return (0.0, "neutral" ) # Register UDF for
                            sentiment analysis sentiment_udf=udf( map_sentiment_udf, StructType([ StructField("polarity", FloatType(), False),
                            StructField("category", StringType(), False) ]) ) def upload_to_s3(local_path: str, s3_key: str, max_retries:
                            int=3): """Upload a file to S3 with exponential backoff retries.""" s3_client=boto3.client('s3') for attempt in
                            range(max_retries): try: s3_client.upload_file(local_path, BUCKET_NAME, s3_key) logger.info(f"Uploaded to
                            s3://{BUCKET_NAME}/{s3_key}") return except Exception as e: logger.warning(f"S3 upload attempt {attempt + 1} failed:
                            {e}") if attempt==max_retries - 1: logger.error(f"Failed to upload to s3://{BUCKET_NAME}/{s3_key} after
                            {max_retries} attempts") raise time.sleep(2 ** attempt) def plot_top_words(top_words_df: pd.DataFrame, fraction:
                            float): """Generate and save a bar chart of the top 10 words with their frequencies.""" colors=['#1f77b4', '#2ca02c'
                            , '#d62728' , '#9467bd' , '#8c564b' , '#e377c2' , '#7f7f7f' , '#bcbd22' , '#17becf' , '#ff7f0e' ]
                            fig=go.Figure([go.Bar( x=top_words_df['word'], y=top_words_df['frequency'], text=top_words_df['frequency'],
                            textposition='auto' , marker_color=colors[:len(top_words_df)], marker_line_color='black' , marker_line_width=1.5,
                            opacity=0.85 )]) # Set dynamic title based on fraction title="Top 10 Words for 20% of the Dataset" if fraction==0.2
                            else f"Top 10 Words for {int(fraction*100)}% of the Dataset" if fraction==1.0:
                            title="Top 10 Words for the Whole Dataset" fig.update_layout( title=title, xaxis_title="Word" ,
                            yaxis_title="Frequency" , title_x=0.5, height=600, width=800, font=dict(family="Arial, sans-serif" , size=16,
                            color="black" ), plot_bgcolor='rgba(240, 240, 240, 0.95)' , paper_bgcolor='white' , xaxis=dict(tickangle=45,
                            gridcolor='lightgray' , title_font=dict(size=18), tickfont=dict(size=14)), yaxis=dict(gridcolor='lightgray' ,
                            title_font=dict(size=18), tickfont=dict(size=14)), showlegend=False, margin=dict(l=50, r=50, t=100, b=100) ) # Save
                            plot to temporary file and upload to S3 path=f"/tmp/top_words_fraction_{fraction:.2f}.png" fig.write_image(path,
                            format="png" , scale=2) upload_to_s3(path, f"{KEYWORD_PLOT_PATH}top_words_fraction_{fraction:.2f}.png") def
                            plot_sentiment_categories(category_counts: List[Tuple[str, int]], fraction:
                            float): """Generate and save a pie chart of sentiment category counts.""" df=pd.DataFrame(category_counts,
                            columns=['category', 'count' ]) colors=['#1f77b4', '#2ca02c' , '#d62728' ] # Colors for positive, neutral, negative
                            fig=go.Figure([go.Pie( labels=df['category'], values=df['count'], textinfo='label+percent' , textposition='inside' ,
                            marker=dict(colors=colors[:len(df)], line=dict(color='black' , width=1.5)), opacity=0.85 )])
                            title="Sentiment Analysis for 20% of the Dataset" if fraction==0.2 else f"Sentiment Analysis for
                            {int(fraction*100)}% of the Dataset" if fraction==1.0: title="Sentiment Analysis for the Whole Dataset"
                            fig.update_layout( title=title, title_x=0.5, height=600, width=800, font=dict(family="Arial, sans-serif" , size=16,
                            color="black" ), plot_bgcolor='rgba(240, 240, 240, 0.95)' , paper_bgcolor='white' , showlegend=True,
                            legend=dict(title="Sentiment Categories" , font=dict(size=14)), margin=dict(l=50, r=50, t=100, b=100) )
                            path=f"/tmp/categories_fraction_{fraction:.2f}.png" fig.write_image(path, format="png" , scale=2) upload_to_s3(path,
                            f"{SENTIMENT_PLOT_PATH}categories_fraction_{fraction:.2f}.png") def run_experiment(spark: SparkSession, fraction:
                            float=0.8): """Run sentiment analysis and keyword extraction on a fraction of the dataset.""" logger.info(f"Running
                            with fraction {fraction:.2f}") metrics: Dict[str, Union[float, int]]={"fraction": fraction} try: # Read and
                            preprocess data start=time.time() df=spark.read.parquet(INPUT_PATH).sample(fraction)
                            df=df.filter(col("cleaned_reviewText").isNotNull() & (trim(col("cleaned_reviewText")) !="" )) # Add dummy row to
                            ensure non-empty DataFrame dummy_row=spark.createDataFrame([("dummy text",)], ["cleaned_reviewText"])
                            df=df.union(dummy_row).repartition(400).cache() metrics["total_records"]=df.count()
                            metrics["read_time_seconds"]=time.time() - start # Keyword extraction using Spark ML keyword_start=time.time()
                            tokenizer=Tokenizer(inputCol="cleaned_reviewText" , outputCol="words" ) df=tokenizer.transform(df)
                            remover=StopWordsRemover(inputCol="words" , outputCol="filtered_words" ) words_df=remover.transform(df)
                            words_exploded=words_df.select(explode(col("filtered_words")).alias("word")).filter(col("word") !="" )
                            word_freq=words_exploded.groupBy("word").count().withColumnRenamed("count", "frequency"
                            ).orderBy(col("frequency").desc()) metrics["keyword_time_seconds"]=time.time() - keyword_start # Save keyword
                            results to S3 output_dir=f"{KEYWORD_OUTPUT_PATH}fraction_{fraction:.2f}/"
                            word_freq.write.mode("overwrite").parquet(output_dir) metrics["keyword_save_time_seconds"]=time.time() -
                            keyword_start # Save top 10 words as JSON top_words_df=word_freq.limit(10).select("word", "frequency" ).toPandas()
                            top_words_json=top_words_df.to_dict(orient='records' ) with open(f"/tmp/top_words_fraction_{fraction:.2f}.json", "w"
                            ) as f: json.dump(top_words_json, f, indent=4) upload_to_s3(f"/tmp/top_words_fraction_{fraction:.2f}.json",
                            f"{KEYWORD_JSON_PATH}top_words_fraction_{fraction:.2f}.json") # Sentiment analysis using TextBlob
                            sentiment_start=time.time() sentiment_df=df.select( col("cleaned_reviewText"),
                            sentiment_udf(col("cleaned_reviewText")).alias("sentiment") ).select( col("cleaned_reviewText"),
                            col("sentiment.polarity").alias("compound"), col("sentiment.category").alias("category")
                            ).filter(col("cleaned_reviewText") !="dummy text" )
                            avg_sentiment=sentiment_df.selectExpr("avg(compound)").collect()[0][0] or 0.0
                            category_counts=sentiment_df.groupBy("category").count().collect() category_counts=[(row["category"], row["count"])
                            for row in category_counts] metrics["sentiment_time_seconds"]=time.time() - sentiment_start
                            metrics["avg_sentiment"]=avg_sentiment for cat, count in category_counts: metrics[f"{cat}_count"]=count # Save
                            sentiment results to S3 output_dir=f"{SENTIMENT_OUTPUT_PATH}fraction_{fraction:.2f}/"
                            sentiment_df.write.mode("overwrite").parquet(output_dir) metrics["sentiment_save_time_seconds"]=time.time() -
                            sentiment_start # Save sentiment categories as JSON category_counts_json=[{"category": cat, "count" : count} for
                            cat, count in category_counts] with open(f"/tmp/categories_fraction_{fraction:.2f}.json", "w" ) as f:
                            json.dump(category_counts_json, f, indent=4) upload_to_s3(f"/tmp/categories_fraction_{fraction:.2f}.json",
                            f"{SENTIMENT_JSON_PATH}categories_fraction_{fraction:.2f}.json") # Generate and save plots
                            plot_top_words(top_words_df, fraction) plot_sentiment_categories(category_counts, fraction) # Calculate final
                            metrics metrics["total_execution_time_seconds"]=time.time() - start
                            metrics["throughput_records_per_second"]=metrics["total_records"] / metrics["total_execution_time_seconds"] if
                            metrics["total_execution_time_seconds"]> 0 else 0
                            metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if
                            metrics["total_records"] > 0 else 0
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
                            """Initialize Spark session and run the experiment for 80% of the dataset."""
                            # Configure Spark session
                            spark = SparkSession.builder \
                            .appName("CombinedAnalysisFraction0.8") \
                            .config("spark.sql.shuffle.partitions", 400) \
                            .config("spark.executor.cores", "4") \
                            .config("spark.executor.memory", "8g") \
                            .config("spark.driver.memory", "8g") \
                            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
                            .config("spark.network.timeout", "600s") \
                            .config("spark.executor.heartbeatInterval", "60s") \
                            .getOrCreate()
                        
                            try:
                            run_experiment(spark, fraction=0.8)
                            except Exception as e:
                            logger.error(f"Error during processing: {e}")
                            raise
                            finally:
                            spark.stop()
                        
                        
                            if __name__ == "__main__":                  main()
                    </code></pre>
                </div>

                <h3 class="text-2xl font-medium text-gray-700 mb-4 mt-6">100percent.py</h3>
                <p class="text-lg text-gray-600 mb-4 leading-relaxed">Run this script (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/100percent.py">100percent.py</a>)
                    for 100% data processing:</p>
                <div class="code-block">
                    <pre><code class="language-python">
                        """Process the entire dataset for sentiment analysis and keyword extraction using Spark."""
                        
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
                        category = 'positive' if polarity > 0.05 else 'negative' if polarity < -0.05 else 'neutral' return (polarity, category)
                            except Exception as e: logger.error(f"Error in sentiment analysis: {e}") return (0.0, "neutral" ) # Register UDF for
                            sentiment analysis sentiment_udf=udf( map_sentiment_udf, StructType([ StructField("polarity", FloatType(), False),
                            StructField("category", StringType(), False) ]) ) def upload_to_s3(local_path: str, s3_key: str, max_retries: int=3)
                            -> None:
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
                            title = "Top 10 Words for 20% of the Dataset" if fraction == 0.2 else f"Top 10 Words for {int(fraction*100)}% of the
                            Dataset"
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
                            colors = ['#1f77b4', '#2ca02c', '#d62728'] # Colors for positive, neutral, negative
                            fig = go.Figure([go.Pie(
                            labels=df['category'],
                            values=df['count'],
                            textinfo='label+percent',
                            textposition='inside',
                            marker=dict(colors=colors[:len(df)], line=dict(color='black', width=1.5)),
                            opacity=0.85
                            )])
                            title = "Sentiment Analysis for 20% of the Dataset" if fraction == 0.2 else f"Sentiment Analysis for
                            {int(fraction*100)}% of the Dataset"
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
                        
                        
                            def run_experiment(spark: SparkSession, fraction: float = 1.0):
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
                            word_freq = words_exploded.groupBy("word").count().withColumnRenamed("count",
                            "frequency").orderBy(col("frequency").desc())
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
                            upload_to_s3(f"/tmp/top_words_fraction_{fraction:.2f}.json",
                            f"{KEYWORD_JSON_PATH}top_words_fraction_{fraction:.2f}.json")
                        
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
                            upload_to_s3(f"/tmp/categories_fraction_{fraction:.2f}.json",
                            f"{SENTIMENT_JSON_PATH}categories_fraction_{fraction:.2f}.json")
                        
                            # Generate and save plots
                            plot_top_words(top_words_df, fraction)
                            plot_sentiment_categories(category_counts, fraction)
                        
                            # Calculate final metrics
                            metrics["total_execution_time_seconds"] = time.time() - start
                            metrics["throughput_records_per_second"] = metrics["total_records"] / metrics["total_execution_time_seconds"] if
                            metrics["total_execution_time_seconds"] > 0 else 0
                            metrics["latency_per_record_ms"] = (metrics["total_execution_time_seconds"] / metrics["total_records"] * 1000) if
                            metrics["total_records"] > 0 else 0
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
                            """Initialize Spark session and run the experiment for the entire dataset."""
                            # Configure Spark session
                            spark = SparkSession.builder \
                            .appName("CombinedAnalysisFraction1.0") \
                            .config("spark.sql.shuffle.partitions", 400) \
                            .config("spark.executor.cores", "4") \
                            .config("spark.executor.memory", "8g") \
                            .config("spark.driver.memory", "8g") \
                            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
                            .config("spark.network.timeout", "600s") \
                            .config("spark.executor.heartbeatInterval", "60s") \
                            .getOrCreate()
                        
                            try:
                            run_experiment(spark, fraction=1.0)
                            except Exception as e:
                            logger.error(f"Error during processing: {e}")
                            raise
                            finally:
                            spark.stop()
                        
                        
                            if __name__ == "__main__":                  main()
                    </code></pre>
                </div>
            </section>

            <section class="bg-white rounded-xl shadow-lg p-8 section-card" aria-labelledby="resources">
                <h2 id="resources" class="text-3xl font-semibold text-gray-800 mb-6 border-b-2 border-indigo-500 pb-2">
                    Additional Resources</h2>
                <p class="text-lg text-gray-600 leading-relaxed">For more details, refer to the sample report (<a
                        href="https://github.com/ShweMoeThantAurum/Scalable-Cloud-Programming-Project/blob/main/Scalable_Cloud_Computing_Report.pdf">Scalable_Cloud_Computing_Report.pdf</a>).
                </p>
            </section>
        </main>
    </div>
</body>

</html>

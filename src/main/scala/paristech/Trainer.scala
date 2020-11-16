package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, feature => P}


object Trainer {

  Logger.getLogger("org").setLevel(Level.ERROR) /* ENLEVE LOGS */

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.master"-> "local",
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()
    import spark.implicits._


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df: DataFrame = spark
      .read
      .parquet("/Users/vince/Documents/Scolarité/Cours_Telecom_Paris/INF729-Hadoop_Spark/TP/cours-spark-telecom/data/prepared_trainingset")
    //df.show(50)

    val tokenizer = new P.RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //val dfTransformed = tokenizer.transform(df)
    //dfTransformed
      //.select("text", "tokens")
      //.show(50)

    val stopWordsRemover = new P.StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    //stopword
      //.transform(dfTransformed)
      //.select("text", "tokens", "filtered")
      //.show(50)

    val countVectorizer = new P.CountVectorizer()
      .setMinDF(65)
      .setMinTF(1)
      .setInputCol("filtered")
      .setOutputCol("tf")

    val idf = new P.IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")

    val countryIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    val currencyIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    val assembler = new P.VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)


    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopWordsRemover,
        countVectorizer,
        idf,
        countryIndexer,
        currencyIndexer,
        assembler,
        lr
      ))

    val pipelineFitted = pipeline.fit(df)
    val dfTransformed = pipelineFitted.transform(df)

    dfTransformed.show(50)


    println("hello world ! from Trainer")

  }
}

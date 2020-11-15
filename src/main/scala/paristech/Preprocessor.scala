package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame,SaveMode,SparkSession}
import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger} /* ENLEVE LOGS */

object Preprocessor {

  Logger.getLogger("org").setLevel(Level.ERROR) /* ENLEVE LOGS */

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.master" -> "local",
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()
    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("/Users/vince/Documents/Scolarité/Cours_Telecom_Paris/INF729-Hadoop_Spark/TP/spark_project_kickstarter_2020_2021/src/train_clean.csv")

    //Affichez le nombre de lignes et le nombre de colonnes dans le DataFrame :
    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    //Affichez un extrait du DataFrame sous forme de tableau :
    df.show()

    //Affichez le schéma du DataFrame, à savoir le nom de chaque colonne avec son type :
    df.printSchema()

    //Assignez le type *Int* aux colonnes qui vous semblent contenir des entiers :
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    //Affichez une description statistique des colonnes de type *Int* :
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    /*Observez les autres colonnes, posez-vous les bonnes questions : quel cleaning faire pour chaque colonne ?
    Y a-t-il des colonnes inutiles ? Comment traiter les valeurs manquantes ?A-t-on des données dupliquées ?
    Quelles sont les valeurs de mes colonnes ? Des répartitions intéressantes ?
    Des "fuites du futur" (vous entendrez souvent le terme *data leakage*) ???
    Proposez des cleanings à faire sur les données : des *groupBy-count*, des *show*, des *dropDuplicates*, etc.*/
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    //Enlevez la colonne *disable_communication*. Cette colonne est très largement majoritairement à *false*, il n'y a que 322 *true* (négligeable), le reste est non-identifié :
    val df2: DataFrame = dfCasted.drop("disable_communication")

    //Ici, pour enlever les données du futur on retire les colonnes *backers_count* et *state_changed_at* :
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    //Il semble y avoir des inversions entre ces deux colonnes et du nettoyage à faire. On remarque en particulier que lorsque `country = "False"` le country à l'air d'être dans currency. On le voit avec la commande
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    /*Créez deux udfs nommées *udf_country* et *udf_currency* telles que :
      - *cleanCountry* : si `country = "False"` prendre la valeur de currency, sinon si country est une chaîne de caractères de taille autre que 2 remplacer par *null*, et sinon laisser la valeur country actuelle. On veut les résultat dans une nouvelle colonne *country2*.
      - *cleanCurrency* : si `currency.length != 3` currency prend la valeur *null*, sinon laisser la valeur currency actuelle. On veut les résultats dans une nouvelle colonne *currency2*.*/
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // ou encore, en utilisant sql.functions.when:
    dfNoFutur
      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      .drop("country", "currency")

    //Changement des valeurs de final_status différentes de 1 par 0
    def formatStatus(status: Integer): Integer = {
      if (status != 1)
        0
      else
        1
    }
    val formatStatusUdf = udf(formatStatus _)

    val dfFinalStatus: DataFrame = dfNoFutur
      .withColumn("final_status", formatStatusUdf($"final_status"))
    dfFinalStatus
      //.withColumn("final_status", formatStatusUdf($"final_status"))
      .groupBy("final_status")
      .count.orderBy($"count".desc)
      .show(50)

    //Ajout de la colonne de durée de campagne et de la colonne des heures de préparation
    val dfAddColumns: DataFrame = dfFinalStatus
      .withColumn("days_campaign",round(($"deadline" - $"launched_at")/(3600*24)).cast("Int"))
      .withColumn("hours_prepa",round(($"launched_at" - $"created_at")/3600).cast("Int"))
      .drop("deadline", "launched_at", "created_at")
    dfAddColumns.show(50)

    //Concaténation des colonne texte
    val dfConcatText: DataFrame = dfAddColumns
      .withColumn("text",concat(lower($"name"), lit(" "), lower($"desc"), lit(" "), lower($"keywords")))
    dfConcatText.show(50)



    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")
  }
}

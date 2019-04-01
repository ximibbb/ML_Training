import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

object tfidfTest {
  def main(args: Array[String]): Unit = {
    val masterUrl = "local"
    val appName = "tfidf_test"
    val sparkConf = new SparkConf().setMaster(masterUrl).setAppName(appName)
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._  //使用DataFrame
    // 读入文档数据
    val docRDD: RDD[(Int, Seq[String])] = sc.parallelize(Seq(
      (0, Array("a", "b", "c","a")),
      (1, Array("c", "b", "b", "c", "a")),
      (2, Array("a", "a", "c","d")),
      (3, Array("c", "a", "b", "a", "a")),
      (4, Array("我", "爱", "旅行", "土耳其", "大理","云南")),
      (5, Array("我", "爱", "学习")),
      (6, Array("胡歌", "优秀","演员", "幽默", "责任感"))
    ))
    val docDF = docRDD.map(x => (x._1, x._2)).toDF("index", "words")
    docDF.show(truncate=false)
    System.out.println("\nThis is OriginalData...")
    // 哈希编码，统计词频
    val hashModel = new HashingTF()
      .setInputCol("words")
      .setOutputCol("hashFeatures")
      .setNumFeatures(2000)

    val hashedData = hashModel.transform(docDF)
    hashedData.show(truncate=false)
    System.out.println("\nThis is HashedData...")
    //计算idf
    val idf = new IDF()
      .setInputCol("hashFeatures")
      .setOutputCol("features")
    val idfModel = idf.fit(hashedData)
    val tfidfData = idfModel.transform(hashedData)
    tfidfData.show(truncate=false)
    System.out.println("\nThis is TfidfData...")
  }
}

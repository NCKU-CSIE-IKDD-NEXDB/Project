from pyspark import SparkContext
from pyspark.sql import *
from pyspark.mllib.classification import *
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
import pandas as pd
import numpy as np

sc = SparkContext()
sqlContext = SQLContext(sc)
lines = sc.textFile("s3n://kaggle-frank/train.csv")
header = lines.first()
data = lines.filter(lambda line: line != header)
parts = data.map(lambda line: line.split(","))

train = parts.map(lambda p: (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19], p[20], p[21], p[22], p[23]))

schemaString = "id click hour C1 banner_pos site_id site_domain site_category app_id app_domain app_category device_id device_ip device_model device_type device_conn_type C14 C15 C16 C17 C18 C19 C20 C21"

fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]

schema = StructType(fields)
schemaTrain = sqlContext.applySchema(train, schema)
schemaTrain.registerTempTable("train")

lines = sc.textFile("s3n://kaggle-frank/test.csv")
header = lines.first()
data = lines.filter(lambda line: line != header)
parts = data.map(lambda line: line.split(","))

test = parts.map(lambda p: (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19], p[20], p[21], p[22]))

schemaString = "id hour C1 banner_pos site_id site_domain site_category app_id app_domain app_category device_id device_ip device_model device_type device_conn_type C14 C15 C16 C17 C18 C19 C20 C21"

fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]

schema = StructType(fields)
schemaTest = sqlContext.applySchema(test, schema)
schemaTest.registerTempTable("test")

results = sqlContext.sql("SELECT click, hour, banner_pos, C1, C14, C15 FROM train")
train_data = results.map(lambda row: LabeledPoint(row.click, [row.hour, row.banner_pos, row.C1, row.C14, row.C15]))

svm = SVMWithSGD.train(train_data)

results = sqlContext.sql("SELECT hour, banner_pos, C1, C14, C15 FROM test")
test_data = results.map(lambda row: [row.hour, row.banner_pos, row.C1, row.C14, row.C15])

ans_rdd = svm.predict(test_data)
ans_list = ans_rdd.collect()

results = sqlContext.sql("SELECT id FROM test")
id_rdd = results.map(lambda row: row.id)
id_list = id_rdd.collect()

ans_np = np.array(ans_list).astype(float)
id_np = np.array(id_list)
fin_ans = pd.DataFrame({'id': id_np, 'click': ans_np})
fin_ans.to_csv('answer.csv', index=False)
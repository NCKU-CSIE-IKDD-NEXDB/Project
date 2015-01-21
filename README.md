# Final Project: Click-Through Rate Prediction

> Predict by Spark & Spot-instance

## Description

* Click-through rate (CTR)
* Click prediction system
* Logarithmic Loss (smaller is better)

## Data

* id
* click
* hour
* banner_pos
* site_id, site_domain, site_category
* app_id, app_domain, app_category
* device_id, device_ip, device_model, device_type, device_conn_type
* C1, C14-21

## Spot instance

* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* -k
* -i
* -s
* --spot-price
* launch, login, destroy

```sh
$ AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx ./spark-ec2 -i {key-pair} -k {key-file-dir} -s {slave-number} --spot-price={price} launch {cluster-name}
```

## Spark

* read csv to sql
* sql
* svm
* Kaggle answer format

## Amazon - s3cmd

### Install

```sh
$ sudo yum --enablerepo epel-testing install s3cmd
```

### Put to s3

```sh
$ AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=xxx s3cmd put answer.csv s3://kaggle-frank
```

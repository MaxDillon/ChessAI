#!/bin/bash

java -cp src/main/resourdes:target/quest-1.0-SNAPSHOT.jar:$(mvn dependency:build-classpath | grep -v INFO) max.dillon.AppKt $1 human prod_model.$1


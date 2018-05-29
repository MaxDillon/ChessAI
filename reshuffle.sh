#!/bin/bash

java -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.ShuffleKt \
     "data.chess2.*.done" shuffled.chess2

java -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.ShuffleKt \
     "data.chess2.*.test" shuffled.chess2 -extension test


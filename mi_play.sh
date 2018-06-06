#!/bin/bash

java -Djava.library.path=$(locate libjnicudnn | sed -e 's/libjnicudnn.so//') -Xmx6g -Xms6g -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.PlayKt $*

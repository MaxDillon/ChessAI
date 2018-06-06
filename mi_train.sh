#!/bin/bash

java -Djava.library.path=$(locate libjnicudnn | sed -e 's/libjnicudnn.so//') -cp target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.TrainKt $*

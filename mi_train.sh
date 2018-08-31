#!/bin/bash

java -Xmx12g -Dorg.bytedeco.javacpp.maxbytes=8g -Dorg.bytedeco.javacpp.maxphysicalbytes=8g -Djava.library.path=/usr/lib/x86_64-linux-gnu -cp target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.TrainKt $*

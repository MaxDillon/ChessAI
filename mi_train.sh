#!/bin/bash

java -Xmx16g -Dorg.bytedeco.javacpp.maxbytes=12g -Dorg.bytedeco.javacpp.maxphysicalbytes=12g -Djava.library.path=/usr/lib/x86_64-linux-gnu -cp target/classes:$(cat .classpath) maximum.industries.TrainKt $*

#!/bin/bash

java -Djava.library.path=jni:${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu -Xmx6g -Xms6g -cp src/main/resources:target/classes:$(cat .classpath) maximum.industries.PlayKt $*

#!/bin/bash

mkdir -p new

for f in shuffled.chess2.*; do  mv $f $(echo $f | sed -e 's/shuffled/data/'); done

java -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.ShuffleKt \
     "data.chess2.*.done" shuffled.chess2

java -cp src/main/resources:target/classes:$(mvn dependency:build-classpath | grep -v INFO) maximum.industries.ShuffleKt \
     "data.chess2.*.test" shuffled.chess2 -extension test

rm data.chess2.000*done

mv data.chess2.*done new

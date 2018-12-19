#!/bin/bash

mkdir -p new

for f in shuffled.chess2.*; do  mv $f $(echo $f | sed -e 's/shuffled/data/'); done

java -cp src/main/resources:target/classes:$(cat .classpath) maximum.industries.ShuffleKt \
     "data.chess2.*.done" shuffled.chess2

java -cp src/main/resources:target/classes:$(cat .classpath) maximum.industries.ShuffleKt \
     "data.chess2.*.test" shuffled.chess2 -extension test

rm data.chess2.000*done
rm data.chess2.000*test

mv data.chess2.*done new
mv data.chess2.*test new

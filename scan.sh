#!/bin/bash

for f in $*; do
    echo "############ $f"
    cat $f | strings | egrep "(Value|Outcome)" \
	| awk '/Value/ {
                   v[n] = $2; n = n + 1
               }
               /Outcome/ {
                   if (n > 20) {
                       if ($2 == "BLACK") {
                           print "1. BLACK"
                       }
                       else if ($2 == "WHITE") {
                           print "7. WHITE"
                       }
                       else if (n < 201) {
                           if (n % 2 == 0) {
                               print "6.   ADVANTAGE_WHITE"
                           } else {
                               print "2.   ADVANTAGE_BLACK"                              
                           }
                       }
                       else {
                           if (v[n-11] > 0.1) {
                               print "5.     DRAW_LEAN_WHITE"
                           } else if (v[n-11] < -0.1) {
                               print "3.     DRAW_LEAN_BLACK"
                           } else {
                               print "5.     DRAW_EVEN"
                           }
                       }
                   } else {
                     print "8. SHORT"
                   }
                   n=0
               }' \
	      | sort | uniq -c | cut -b 1-8,11-
    
done

#!/bin/bash

maxip=192.168.3.240

knock -u $maxip 5312 5479 1830
sftp -P 8022 -b - max@$maxip <<EOF
cd IdeaProjects/quest
-get data.chess2.*.done
-rm data.chess2.*.done
-put $(ls -t model.chess2.Model013.* | head -n 1)
EOF

#paperip=74.82.25.180
#
#sftp -b - paperspace@$paperip <<EOF
#cd quest
#-get data.chess2.*.done
#-rm data.chess2.*.done
#-put $(ls -t model.chess2.* | head -n 1)
#EOF

gceip1=35.233.149.51
gceip2=35.185.203.89

sftp -b - tom_dillon@$gceip1 <<EOF
cd quest
-get data.chess2.*.done
-rm data.chess2.*.done
-put $(ls -t model.chess2.Model013.* | head -n 1)
EOF

sftp -b - tom_dillon@$gceip2 <<EOF
cd quest
-get data.chess2.*.done
-rm data.chess2.*.done
-put $(ls -t model.chess2.Model013.* | head -n 1)
EOF

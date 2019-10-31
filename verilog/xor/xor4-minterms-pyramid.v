0,IN,i0,w0
0,IN,i1,w1
0,IN,i2,w2
0,IN,i3,w3
1,not,noti0,noti0_out,w0
1,not,noti1,noti1_out,w1
1,not,noti2,noti2_out,w2
1,not,noti3,noti3_out,w3

1,and,minterm00,minterm00_out,w0,noti1_out
1,and,minterm01,minterm01_out,noti2_out,noti3_out
1,and,minterm0,minterm0_out,minterm00_out,minterm01_out

1,and,minterm10,minterm10_out,noti0_out,w1
1,and,minterm11,minterm11_out,noti2_out,noti3_out
1,and,minterm1,minterm1_out,minterm10_out,minterm11_out

1,and,minterm20,minterm20_out,noti0_out,noti1_out
1,and,minterm21,minterm21_out,w2,noti3_out
1,and,minterm2,minterm2_out,minterm20_out,minterm21_out

1,and,minterm30,minterm30_out,noti0_out,noti1_out
1,and,minterm31,minterm31_out,noti2_out,w3
1,and,minterm3,minterm3_out,minterm30_out,minterm31_out

1,or,maxterm00,maxterm00_out,minterm0_out,minterm1_out
1,or,maxterm01,maxterm01_out,minterm2_out,minterm3_out
1,or,maxterm02,maxterm02_out,maxterm00_out,maxterm01_out
0,OUT,o,,maxterm02_out

0,IN,i0,w0
0,IN,i1,w1
0,IN,i2,w2
0,IN,i3,w3
1,not,noti0,noti0_out,w0
1,not,noti1,noti1_out,w1
1,not,noti2,noti2_out,w2
1,not,noti3,noti3_out,w3
1,and,minterm0,minterm0_out,w0,noti1_out,noti2_out,noti3_out
1,and,minterm1,minterm1_out,noti0_out,w1,noti2_out,noti3_out
1,and,minterm2,minterm2_out,noti0_out,noti1_out,w2,noti3_out
1,and,minterm3,minterm3_out,noti0_out,noti1_out,noti2_out,w3
1,or,maxterm,maxterm_out,minterm0_out,minterm1_out,minterm2_out,minterm3_out
0,OUT,o,,maxterm_out
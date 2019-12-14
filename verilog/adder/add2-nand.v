0,IN,i0,a
0,IN,i1,b
0,IN,i2,ci

1,nand,u1,u1out,a,b
1,nand,u2,u2out,a,u1out
1,nand,u3,u3out,b,u1out
1,nand,u4,u4out,u2out,u3out
1,nand,u5,u5out,u4out,ci
1,nand,u6,u6out,u4out,u5out
1,nand,u7,u7out,u5out,ci
1,nand,u8,sum,u6out,u7out
1,nand,u9,carryout,u1out,u5out

0,OUT,s,,sum
0,OUT,co,,carryout

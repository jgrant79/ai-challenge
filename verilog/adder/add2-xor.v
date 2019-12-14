0,IN,i0,a
0,IN,i1,b
0,IN,i2,ci

1,xor,xor1,xout1,a,b
1,xor,xor2,sum,xout1,ci
1,and,and1,aout1,ci,xout1
1,and,and2,aout2,a,b
1,or,carryout,aout1,aout2

0,OUT,s,,sum
0,OUT,co,,carryout

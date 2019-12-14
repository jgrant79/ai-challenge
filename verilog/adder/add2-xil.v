0,IN,i0,a
0,IN,i1,b
0,IN,i2,ci

1,not,a_inv,an,a
1,not,b_inv,bn,b
1,not,ci_inv,cin,ci

1,and,and1,mt1,a,bn
1,and,and2,mt2,an,b
1,or,or1,a_xor_b,mt1,mt2
1,not,axb_inv,a_xor_bn,a_xor_b

1,xor,sum,a_xor_b,ci

1,mux2,mux21,carryout,a_xor_b,a,ci

0,OUT,s,,sum
0,OUT,co,,carryout

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

1,xor,xor1,sum,a_xor_b,ci

1,and,and3,mt3,a,a_xor_b
1,and,and4,mt3,ci,a_xor_bn
1,or,or2,carryout,mt5,mt6,mt7

0,OUT,s,,sum
0,OUT,co,,carryout

'''use the table_name that is "01_25c_npy" as the standard
the x-axis is in the order of "02_55c_npy, 03_85c_npy, 04_55c_npy"
TCR is defined as 
    R1=1/G1, R1, G1 @25c

    R2=1/G2, R2, G2 @T (T=55c, 85c)

    ΔR=(R2-R1)=(1/G2-1/G1)=(G1-G2)/(G1G2)

    ΔR/R1=(R2-R1)/R1=(G1-G2)/(G1G2)*G1=(G1-G2)/G2

    TCR=ΔR/R1/ΔT= (G1-G2)/G2/ΔT

    TCR=(G1-G2)/G2/ΔT
the G should be the averge of the group
the result TCR plot should have the number of curves the same to the nunmber of elements in the selected_groups '''
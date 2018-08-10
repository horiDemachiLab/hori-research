# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt

tb_csv = open("H5_Data/PIO/2006.06.01-15/5IM5N11-VbT.csv", "r", encoding="utf_8")
reader = csv.reader(tb_csv)
header = next(reader)
lista = []
for row in reader:
    lista.append(row[3])
    
plt.plot(range(4752), lista)
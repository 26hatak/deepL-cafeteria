import csv
import os
import shutil

with open("daily_sort.csv", "r") as file:
    spamreader = csv.reader(file)
    for day in spamreader:
        total = day[3]
        p1 = int(day[4])/int(total)
        p2 = int(day[5])/int(total)
        p3 = int(day[6])/int(total)
        print(p1, p2, p3)
#!/usr/bin/env python3

import sys

salesTotal = 0
oldKey = None

# Loop around the data
# It will be in the format key\tval
# Where key is the store name, val is the sale amount
#
# All the sales for a particular store will be presented,
# then the key will change and we'll be dealing with the next store

salesTotal = 0
numOfSales = 0
for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    if len(data_mapped) != 2:
        # Something has gone wrong. Skip this line.
        continue

    thisKey, thisSale = data_mapped

    oldKey = thisKey
    salesTotal += float(thisSale)
    numOfSales += 1

print(numOfSales, "\t", round(salesTotal,2))

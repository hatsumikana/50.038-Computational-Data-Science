#!/usr/bin/env python3

import sys

oldKey = None
totalOccurrences = 0
highestOccurrences = 0

for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    if len(data_mapped) != 2:
        # Something has gone wrong. Skip this line.
        continue

    thisKey, thisStatus = data_mapped

    if oldKey and oldKey != thisKey:
        if totalOccurrences >= highestOccurrences:
            highest_visited_path = oldKey
            highestOccurrences = totalOccurrences

        oldKey = thisKey;
        totalOccurrences = 0

    oldKey = thisKey
    totalOccurrences += 1

print(os.path.basename(highest_visited_path), "\t", highestOccurrences)
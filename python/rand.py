#!/usr/bin/env python

import random

in_file = open("../input/test.csv", 'r')
out_file = open("../out.csv", 'w')

in_file.readline() # ignore column headers
out_file.write("id,target\n")

for line in in_file:
    line_id = line.split(",")[0]
    output = "1" if (random.random() < 0.5) else "0"
    out_file.write(line_id + "," + output + "\n")

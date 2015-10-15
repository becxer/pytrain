#!/usr/bin/python
#(filename and content) to (answer 1024 digit)

import sys
import os

if len(sys.argv) < 3:
	print "usage ./img2Vec.py dir-path target-file.txt"
	exit()

path = sys.argv[1]
trgfname  = sys.argv[2]

print path, trgfname
dirs = os.listdir(path)
print dirs

tf = open(trgfname,'w')
for fname in dirs:
	first = fname.split('_')[0] 
	f = open(path + '/' + fname)
	fr = list(f.read())
	features = [x for x in fr if x != '\r' and x != '\n']
	features.insert(0,first)
	print features
	f.close()
	tf.write(fname+'\t')
	for c in features:
		tf.write(c + '\t')
	tf.write('\n')

tf.close()

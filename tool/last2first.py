#!/usr/bin/python
import sys

if len(sys.argv) is 2:
  print "usage : last2first.py src trg"
  exit()

src = sys.argv[1]
trg = sys.argv[2]
print src + " -> " + trg

fsrc = open(src)
ftrg = open(trg,'w')
for line in fsrc.readlines():
	line = line.strip()
	line = line.split('\t')
	lsize = len(line)
	res = line[lsize -1]
	for i in range(len(line)-1):
		res += '\t' + line[i]
	ftrg.write(res+'\n')

fsrc.close()
ftrg.close()

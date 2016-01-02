import os
import re

outf = open("./email_word2.txt","w")

for fname in os.listdir("./"):
    if os.path.isdir(fname):
        label = fname
        print label
        for content in os.listdir("./"+label):
            print content
            c1 = open("./"+label+"/"+content)
            c2 = c1.read().replace('\n',' ').replace('\r',' ')
            c3 = re.compile("([\w][\w]*'?\w?)").findall(c2)
            c1.close()

            line = str(label)
            for word in c3:
                line += "\t" + word
            print line
            outf.write(line+"\n")

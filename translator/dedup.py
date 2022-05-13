
f1 = open("bx_by.tsv", "r")
f2 = open("unshuffled_bx.txt", "r")
a2 = f2.readlines()
a1 = f1.readlines()
dic = {}

for line in a1:
#	print(strs)
	strs = line.split("\t")
	print(strs)
	dic[strs[0].strip()] = strs[1].strip()

f3 = open("dedup_by", "w")

for line in a2:
	if line.strip() in dic:
		f3.write(dic[line.strip()])
	f3.write("\n")


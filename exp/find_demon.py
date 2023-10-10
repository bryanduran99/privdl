# find all unique shapes in a.txt, 
shapes = set() # 使用set记录出现过的所有形状
with open('a.txt') as f:
    for line in f:
        shape = line.split()[-1] # 提取形状
        shapes.add(shape) # 将形状添加到集合中
for shape in shapes:
    print(shape)
    


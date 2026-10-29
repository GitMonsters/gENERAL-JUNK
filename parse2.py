import json, re
with open('qcompile-optimizer.log') as f:
    s = f.read()
# Find all occurrences of data in stdout blocks
# Better to just search for paths
for m in re.findall(r'(/kaggle/input/[^\s"\\]+)', s):
    print(m)

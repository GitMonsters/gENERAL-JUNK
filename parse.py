import json, re
with open('qcompile-optimizer.log') as f:
    s = f.read()
# Try to find all stdout data
for m in re.finditer(r'"stream_name":"stdout".*?"data":"(.*?)"', s):
    print(m.group(1))
    break
# Also look for any data with kaggle paths
for m in re.findall(r'data":"([^"]+/kaggle[^"]*)"', s):
    print(m)

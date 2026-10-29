import re
with open('qcompile-optimizer.log') as f:
    s = f.read()
# Get all stdout data
for m in re.finditer(r'"stream_name":"stdout","time":[^,]+,"data":"(.*?)"', s):
    print(m.group(1))

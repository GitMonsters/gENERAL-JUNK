import re
with open('qcompile-optimizer.log') as f:
    s = f.read()
# Look for module errors
idx = s.find('ModuleNotFoundError')
if idx >= 0:
    start = max(0, idx-50)
    end = min(len(s), idx+400)
    print(s[start:end])
else:
    print('none')

import re
with open('qcompile-optimizer.log') as f:
    s = f.read()
# Look for exceptions
idx = s.find('Exception')
if idx >= 0:
    # Get surrounding
    start = max(0, idx-100)
    end = min(len(s), idx+800)
    print(s[start:end])
else:
    print('none')

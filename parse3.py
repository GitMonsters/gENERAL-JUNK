import re
with open('qcompile-optimizer.log') as f:
    s = f.read()
# Look for directory listings
for m in re.findall(r'\[([^\]]*)\]', s):
    if 'google-verifiable' in m or 'qcompile' in m or 'evanpieser' in m:
        print(m)

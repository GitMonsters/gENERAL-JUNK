import json
with open('qcompile-optimizer.log') as f:
    content = f.read()
for line in content.splitlines():
    if '"stream_name":"stdout"' in line:
        try:
            obj = json.loads(line)
            print(obj['data'], end='')
        except:
            pass

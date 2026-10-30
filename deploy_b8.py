import json, subprocess, sys, os, base64, hashlib
BIN = "/opt/hatch/bin/vercel"
def call(tool, payload):
    r = subprocess.run([BIN, "call-tool", "--name", tool, "--arguments-json", json.dumps(payload)], capture_output=True, text=True)
    try: return json.loads(r.stdout)
    except Exception: print((r.stdout or "")[-600:]); sys.exit(1)
for p in ["models/m_tabular6.json", "models/m_softmax8.json"]:
    data = open(p, "rb").read()
    res = call("upload_file", {"requestBody": base64.b64encode(data).decode(), "xVercelDigest": hashlib.sha1(data).hexdigest(), "contentLength": len(data)})
    print("uploaded", p, "ERR" if '"isError": true' in json.dumps(res) else "ok")
files = [{"file": "index.html", "data": open("index.html", encoding="utf-8").read()}]
for f in sorted(os.listdir("models")):
    p = os.path.join("models", f)
    files.append({"file": p, "sha": hashlib.sha1(open(p, "rb").read()).hexdigest(), "size": os.path.getsize(p)})
res = call("create_deployment", {"projectId": "prj_FryL6G5sqdFSzSU7Jg0XebhPqOp8",
    "requestBody": {"name": "ai-projects-app", "files": files, "projectSettings": {"framework": None}, "target": "production"}})
print(json.dumps(res)[:250])

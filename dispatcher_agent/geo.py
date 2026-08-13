import asyncio, json, urllib.request, base64
import websockets
def target():
    ts=json.load(urllib.request.urlopen("http://127.0.0.1:9222/json"))
    return next(t for t in ts if t["type"]=="page")["webSocketDebuggerUrl"]
async def main():
    async with websockets.connect(target(), max_size=200_000_000) as ws:
        i=0
        async def cmd(m,**p):
            nonlocal i; i+=1
            await ws.send(json.dumps({"id":i,"method":m,"params":p}))
            while True:
                r=json.loads(await ws.recv())
                if r.get("id")==i: return r.get("result",{})
        async def js(e):
            r=await cmd("Runtime.evaluate",expression=f"(()=>{{{e}}})()",returnByValue=True)
            return r["result"].get("value")
        await asyncio.sleep(3)
        print(json.dumps(await js("""
          const g=[...document.querySelectorAll('.jbox')].slice(0,5).map(b=>{
            const r=b.getBoundingClientRect();
            return {job:b.dataset.job, sev:[...b.classList].find(c=>c.startsWith('sev-')),
                    x:Math.round(r.x), y:Math.round(r.y+window.scrollY),
                    w:Math.round(r.width), bottom:Math.round(r.bottom+window.scrollY)};
          });
          return g;""")))
asyncio.run(main())

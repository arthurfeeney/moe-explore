import json
import matplotlib.pyplot as plt

with open("trace.json") as f:
    trace = json.load(f)

trace_events = trace["traceEvents"]

kernels = [e for e in trace_events if "cat" in e and e["cat"] == "kernel"]
kernels = sorted(kernels, key=lambda k: float(k["ts"]))

start_time = kernels[0]["ts"]
for k in kernels:
    k["ts"] -= start_time
    k["te"] = k["ts"] + k["dur"]

lines = [
    [(k["ts"], 1), (k["te"], 1)]


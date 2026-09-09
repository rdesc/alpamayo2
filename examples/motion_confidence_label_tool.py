# SPDX-License-Identifier: Apache-2.0
"""Local web app for manually labeling self-generated CoC correctness: Correct (Yes) /
Incorrect (No) / Unsure, one candidate at a time, with the same reduced 3-camera/1-frame
images used for Claude's own vision-labeling pass (Addendum 2 in
docs/motion_confidence_experiment.md) -- so human labels land in the same schema and are
directly comparable to the existing Qwen v1/v2 and Claude verdicts.

Pure stdlib (http.server), no new dependencies. Labels are saved to disk after every click
(atomic write), so the tool is safe to stop and resume at any time -- it always reopens on
the first unlabeled item.

Usage
-----
    python examples/motion_confidence_label_tool.py \\
        --manifest outputs/motion_confidence_images/front3_qa/manifest.json \\
        --out outputs/motion_confidence_human_labels.json \\
        --port 8899

Then open http://localhost:8899 in a browser (port-forward if running on a remote
machine). Keyboard shortcuts: Y = Correct, N = Incorrect, U = Unsure, Left/Right = Prev/Next
(navigate without changing an existing label).

Output schema (outputs/motion_confidence_human_labels.json), matching
outputs/motion_confidence_claude_vision_labels_n10.json:
    [{"index": int, "clip_id": str, "t0_us": int,
      "labels": [{"candidate_index": int, "self_coc": str, "verdict": "Yes"|"No"|"Unsure"}]}]
"""

import argparse
import html
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERDICTS = {"yes": "Yes", "no": "No", "unsure": "Unsure"}


def load_items(manifest_path):
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    items = []
    for ev in manifest:
        for ci, coc in enumerate(ev["self_cocs"]):
            items.append({
                "index": ev["index"],
                "clip_id": ev["clip_id"],
                "t0_us": ev["t0_us"],
                "candidate_index": ci,
                "self_coc": coc,
                "image_paths": ev["image_paths"],
            })
    return items


class LabelStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        if os.path.exists(path):
            self.events = {ev["index"]: ev for ev in json.load(open(path, encoding="utf-8"))}
        else:
            self.events = {}

    def get_verdict(self, index, candidate_index):
        ev = self.events.get(index)
        if not ev:
            return None
        for lab in ev["labels"]:
            if lab["candidate_index"] == candidate_index:
                return lab["verdict"]
        return None

    def set_verdict(self, item, verdict):
        with self.lock:
            ev = self.events.setdefault(item["index"], {
                "index": item["index"], "clip_id": item["clip_id"], "t0_us": item["t0_us"], "labels": [],
            })
            for lab in ev["labels"]:
                if lab["candidate_index"] == item["candidate_index"]:
                    lab["verdict"] = verdict
                    break
            else:
                ev["labels"].append({
                    "candidate_index": item["candidate_index"],
                    "self_coc": item["self_coc"],
                    "verdict": verdict,
                })
            self._save()

    def counts(self):
        n = {"Yes": 0, "No": 0, "Unsure": 0}
        for ev in self.events.values():
            for lab in ev["labels"]:
                n[lab["verdict"]] += 1
        return n

    def _save(self):
        out = [self.events[k] for k in sorted(self.events)]
        for ev in out:
            ev["labels"].sort(key=lambda r: r["candidate_index"])
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, self.path)


PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>CoC labeling</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 20px; }}
.progress {{ margin-bottom: 12px; font-size: 14px; color: #aaa; }}
.imgs {{ display: flex; gap: 8px; }}
.imgs img {{ width: 32%; border: 1px solid #333; border-radius: 4px; }}
.coc {{ font-size: 22px; margin: 20px 0; padding: 16px; background: #1c1c1c; border-radius: 8px; }}
.meta {{ font-size: 12px; color: #777; margin-bottom: 8px; }}
.buttons {{ display: flex; gap: 12px; margin: 16px 0; }}
button {{ font-size: 18px; padding: 14px 28px; border-radius: 8px; border: none; cursor: pointer; }}
.yes {{ background: #2e7d32; color: white; }}
.no {{ background: #c62828; color: white; }}
.unsure {{ background: #616161; color: white; }}
button.active {{ outline: 4px solid #fff; }}
.nav {{ margin-top: 20px; }}
.nav a {{ color: #8ab4f8; margin-right: 16px; text-decoration: none; }}
</style></head>
<body>
<div class="progress">Labeled {n_labeled}/{n_total} &nbsp;|&nbsp; Yes: {n_yes} No: {n_no} Unsure: {n_unsure}</div>
<div class="meta">event {index} (pos {pos}/{n_total}), candidate {candidate_index} &mdash; clip {clip_id} t0={t0_us}</div>
<div class="imgs">
<img src="/img?pos={pos}&slug=front_left">
<img src="/img?pos={pos}&slug=front_wide">
<img src="/img?pos={pos}&slug=front_right">
</div>
<div class="coc">{self_coc}</div>
<form method="post" action="/label" id="f">
<input type="hidden" name="pos" value="{pos}">
<div class="buttons">
<button type="submit" name="verdict" value="yes" class="yes {yes_active}">Correct (Y)</button>
<button type="submit" name="verdict" value="no" class="no {no_active}">Incorrect (N)</button>
<button type="submit" name="verdict" value="unsure" class="unsure {unsure_active}">Unsure (U)</button>
</div>
</form>
<div class="nav">
<a href="/?pos={prev_pos}">&larr; Prev</a>
<a href="/?pos={next_pos}">Next &rarr;</a>
<a href="/?pos={first_unlabeled}">Jump to first unlabeled</a>
</div>
<script>
document.addEventListener('keydown', function(e) {{
  var m = {{y: 'yes', n: 'no', u: 'unsure'}};
  var k = e.key.toLowerCase();
  if (m[k]) {{ document.querySelector('button[value="' + m[k] + '"]').click(); }}
  if (e.key === 'ArrowLeft') {{ window.location.href = '/?pos={prev_pos}'; }}
  if (e.key === 'ArrowRight') {{ window.location.href = '/?pos={next_pos}'; }}
}});
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    items = None  # set on the class before serving
    store = None

    def log_message(self, fmt, *args):
        pass  # keep stdout quiet

    def _first_unlabeled(self):
        for i, it in enumerate(self.items):
            if self.store.get_verdict(it["index"], it["candidate_index"]) is None:
                return i
        return len(self.items) - 1

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path == "/img":
            pos = int(qs["pos"][0])
            slug = qs["slug"][0]
            item = self.items[pos]
            img_path = item["image_paths"][slug]
            if not os.path.isfile(img_path):
                self.send_error(404)
                return
            with open(img_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/":
            n_total = len(self.items)
            pos = int(qs.get("pos", [self._first_unlabeled()])[0])
            pos = max(0, min(pos, n_total - 1))
            item = self.items[pos]
            verdict = self.store.get_verdict(item["index"], item["candidate_index"])
            counts = self.store.counts()
            n_labeled = counts["Yes"] + counts["No"] + counts["Unsure"]
            body = PAGE_TEMPLATE.format(
                n_labeled=n_labeled, n_total=n_total,
                n_yes=counts["Yes"], n_no=counts["No"], n_unsure=counts["Unsure"],
                index=item["index"], pos=pos, candidate_index=item["candidate_index"],
                clip_id=item["clip_id"], t0_us=item["t0_us"],
                self_coc=html.escape(item["self_coc"]),
                yes_active="active" if verdict == "Yes" else "",
                no_active="active" if verdict == "No" else "",
                unsure_active="active" if verdict == "Unsure" else "",
                prev_pos=max(0, pos - 1), next_pos=min(n_total - 1, pos + 1),
                first_unlabeled=self._first_unlabeled(),
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/label":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)
        pos = int(form["pos"][0])
        verdict_key = form["verdict"][0]
        verdict = VERDICTS[verdict_key]
        item = self.items[pos]
        self.store.set_verdict(item, verdict)
        next_pos = min(pos + 1, len(self.items) - 1)
        self.send_response(303)
        self.send_header("Location", f"/?pos={next_pos}")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="outputs/motion_confidence_images/front3_qa/manifest.json")
    parser.add_argument("--out", default="outputs/motion_confidence_human_labels.json")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()

    Handler.items = load_items(args.manifest)
    Handler.store = LabelStore(args.out)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Loaded {len(Handler.items)} candidates from {args.manifest}")
    print(f"Labels save to {args.out} (resumable)")
    print(f"Open http://localhost:{args.port} in a browser")
    print("Keyboard: Y=Correct, N=Incorrect, U=Unsure, Left/Right=Prev/Next")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

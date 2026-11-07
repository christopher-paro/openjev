"""Run decisions off the event loop, up to QEV_PARALLEL at a time.

Upstream calls the blocking engine.decide from the async handler, so one request stops every other request and the
health endpoint until it finishes. Threads share the one loaded model: a short decision can finish while a long one runs.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
src = path.read_text()


def replace(old: str, new: str) -> None:
    global src
    if src.count(old) != 1:
        sys.exit(f"expected exactly one {old!r} in {path}")
    src = src.replace(old, new)


replace(
    "            response, meta = engine.decide(req, debias=debias, debug=debug)\n",
    "            async with decide_slots:\n"
    "                response, meta = await run_in_threadpool(engine.decide, req, debias=debias, debug=debug)\n",
)
replace(
    '    @app.post("/v1/systemone")\n',
    '    decide_slots = asyncio.Semaphore(int(os.environ.get("QEV_PARALLEL", "4")))\n\n'
    '    @app.post("/v1/systemone")\n',
)
src = "import asyncio\nimport os\n\nfrom starlette.concurrency import run_in_threadpool\n" + src if not src.startswith(('"""', "from __future__")) else src
if "run_in_threadpool\n" not in src.split("decide_slots")[0]:
    marker = "\nimport " if "\nimport " in src else "\nfrom "
    head, tail = src.split(marker, 1)
    src = head + "\nimport asyncio\nimport os\nfrom starlette.concurrency import run_in_threadpool" + marker + tail
path.write_text(src)

# mammoth-1's config carries vision_config, so the CLI would pick MMDecisionEngine, which serve.patch does not touch:
# the model's readout and block attention would be skipped. Only an explicit --multimodal takes that engine.
cli = path.with_name("cli.py")
text = cli.read_text()
old = 'if args.multimodal or config.get("vision_config") is not None:'
if text.count(old) != 1:
    sys.exit(f"expected exactly one {old!r} in {cli}")
cli.write_text(text.replace(old, "if args.multimodal:"))

"""Answer a single-question request in one forward pass instead of a prefill plus a branch.

Every forward costs about 270 ms on the Spark however short the input, so the two passes are most of a short decision.
With one question nothing is shared, so the prefix cache buys nothing. At load the engine compares the two paths and
keeps the prefill path if they disagree, then runs one long request so the first real one does not pay for it.
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
    '        log.info("loaded %s (%s) in %.1fs, fork_mode=%s", model_path,',
    '        self.one_pass = False\n'
    '        self._select_one_pass()\n'
    '        self._warm()\n'
    '        log.info("loaded %s (%s) in %.1fs, fork_mode=%s", model_path,',
)
replace(
    '        if self.info.fork_mode == "batched":\n            try:\n                branches = self._run_batched(',
    '        if self.one_pass and len(rendered) == 1:\n'
    '            branches = self._run_single(request.state, rendered)\n'
    '        elif self.info.fork_mode == "batched":\n            try:\n                branches = self._run_batched(',
)
replace(
    '    def _split(self, state: Any, rendered: list[RenderedQuestion]):\n',
    '''    def _run_single(self, state: Any, rendered: list[RenderedQuestion]) -> list[BranchResult]:
        import torch

        prefix_ids, suffixes, _ = self._split(state, rendered)
        r, suf = rendered[0], suffixes[0]
        P, n = len(prefix_ids), len(prefix_ids) + len(suf)
        with torch.inference_mode(), self._blocks([P], [n], n):
            o = self.model(input_ids=torch.tensor([prefix_ids + suf], device=self.device),
                           attention_mask=torch.ones((1, n), dtype=torch.long, device=self.device),
                           use_cache=False, return_dict=True, logits_to_keep=1)
            logits, mass = self._readout(o.logits[0, -1], r.n_slots)
        return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(suf))]

    def _probe_questions(self) -> list[RenderedQuestion]:
        from .schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
        qs = {
            "a": NoulQuestion(type="noul", instructions="Is the customer asking for money back?"),
            "b": ChoiceQuestion(type="choice", instructions="Which team should handle this?",
                                criteria={"billing": "charges and refunds", "technical": "bugs", "sales": "pricing"}),
            "c": ScoreQuestion(type="score", instructions="How frustrated is the customer?",
                               criteria=["calm", "irritated", "angry"]),
        }
        return [render_question(k, q, style=self.style) for k, q in qs.items()]

    def _select_one_pass(self, tolerance: float = 0.01) -> None:
        if self.readout.head is not None:
            log.warning("one pass needs the slot readout; keeping the prefill path")
            return
        states = [{"ticket": "My card was charged twice for the same order and support has not replied in 3 days."},
                  "Fix the typo in the README heading.",
                  "The deploy fails after the migration runs. " * 100]
        try:
            diff = 0.0
            for state in states:
                for r in self._probe_questions():
                    two = self._run_sequential(state, [r])[0]
                    one = self._run_single(state, [r])[0]
                    diff = max(diff, max(abs(a - b) for a, b in zip(softmax(two.logits), softmax(one.logits))))
        except Exception as exc:
            log.warning("one pass unavailable (%s); keeping the prefill path", exc)
            return
        self.one_pass = diff <= tolerance
        self.info.extra["one_pass_max_diff"] = float(diff)
        log.info("one pass %s: max probability difference vs prefill %.5f", "on" if self.one_pass else "off", diff)

    def _warm(self) -> None:
        t0 = time.perf_counter()
        state = "word " * min(6000, self.max_branch_tokens - 512)
        rendered = self._probe_questions()
        if self.one_pass:
            self._run_single(state, rendered[:1])
        if self.info.fork_mode == "batched":
            self._run_batched(state, rendered)
        else:
            self._run_sequential(state, rendered)
        log.info("warmed in %.1fs", time.perf_counter() - t0)

    def _split(self, state: Any, rendered: list[RenderedQuestion]):
''',
)
path.write_text(src)

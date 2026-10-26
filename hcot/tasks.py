"""State-tracking task generators with exact answers.

T1 reassign  last write wins over 4 variables (constants only: a counter-free flip-flop)
T2 switch    parity of toggles
T3 cups      3-cup shell game, transpositions of S3 acting on the ball's cup
T4 boxes     5 items in 5 boxes, transpositions of S5, query one box
T5 switches  4 named switches, each step toggles one of them, query one switch at the end
             (parity spread over 4 bits: Z2^4; the queried switch is only named after the steps)

Each task samples a latent instance (initial state, steps, query) and renders it as natural
language ("nl") or compact symbols ("sym"). An item is a dict with the user message split into
parts (``prefix`` = instructions + initial state, ``steps``, ``fill``, ``question``) so token
spans can be located after templating, plus ``answer``, ``cands`` (every possible answer),
``trace`` (the queried value after every step, index 0 = initial) and ``full`` (the whole final
state: every variable / box).

``pair`` makes a counterfactual twin that differs only in the initial state (one value), so the
steps' tokens are identical and the final answer differs: used for state patching.

Why T1 has no copies ("y = x"): with copies among 3+ variables a temp-swap realizes a 2-cycle, so
the task's transition monoid is no longer aperiodic and parity embeds in it. Constants only keeps
T1 on the counter-free side that [0, 1] eigenvalues can handle.
"""

import copy
import random
import re

LENGTHS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
TASKS = ["T1", "T2", "T3", "T4"]
FORMATS = ["nl", "sym"]
EXTRA_TASKS = ["T5"]  # follow-up 3; not in TASKS so the P1 defaults are unchanged
CHANCE = {"T1": 0.1, "T2": 0.5, "T3": 1 / 3, "T4": 0.2, "T5": 0.5}

VARS = ["a", "b", "c", "d"]
NAMES = ["Ann", "Ben", "Cal", "Dee", "Eve", "Fay", "Gus", "Hal", "Ida", "Jo"]
CUPS = ["A", "B", "C"]
ITEMS = ["apple", "book", "cat", "drum", "egg"]
SYM_ITEMS = ["a", "b", "c", "d", "e"]
SWITCHES = ["a", "b", "c", "d"]

FILLER = ("The weather that day was mild, and a few clouds drifted slowly over the hills while "
          "birds sang in the trees near the old stone bridge. ")


def filler_text(n_sent: int) -> str:
    return "" if n_sent == 0 else "\n" + FILLER * n_sent + "\n"


# --- latent samplers ---------------------------------------------------------------------------


def s1(rng, n):
    init = {v: rng.randrange(10) for v in VARS}
    q = rng.choice(VARS)
    p = rng.randrange(n + 1)  # step of q's last write (0 = the initial value)
    steps = []
    for i in range(1, n + 1):
        if i == p:
            v = q
        else:
            v = rng.choice(VARS if i < p else [x for x in VARS if x != q])
        steps.append((v, rng.randrange(10)))
    return dict(init=init, steps=steps, q=q, names=[])


def s2(rng, n):
    return dict(init=int(rng.random() < 0.5), steps=[rng.random() < 0.5 for _ in range(n)], q=None,
                names=[rng.choice(NAMES) for _ in range(n)])


def s3(rng, n):
    steps = []
    for _ in range(n):
        i, j = rng.sample(range(3), 2)
        steps.append((i, j))
    return dict(init=rng.randrange(3), steps=steps, q=None, names=[])


def s4(rng, n):
    init = list(range(5))
    rng.shuffle(init)  # init[box] = item
    return dict(init=init, steps=[tuple(rng.sample(range(5), 2)) for _ in range(n)],
                q=rng.randrange(5), names=[])


def s5(rng, n):
    return dict(init=[rng.randrange(2) for _ in SWITCHES], steps=[rng.randrange(len(SWITCHES)) for _ in range(n)],
                q=rng.randrange(len(SWITCHES)), names=[rng.choice(NAMES) for _ in range(n)])


# --- semantics ----------------------------------------------------------------------------------


def run(task, lat):
    """(trace of the queried value, full final state)."""
    st = copy.deepcopy(lat["init"])
    get = {
        "T1": lambda s: s[lat["q"]],
        "T2": lambda s: s,
        "T3": lambda s: s,
        "T4": lambda s: s[lat["q"]],
        "T5": lambda s: s[lat["q"]],
    }[task]
    trace = [get(st)]
    for x in lat["steps"]:
        if task == "T1":
            st[x[0]] = x[1]
        elif task == "T2":
            st ^= int(x)
        elif task == "T3":
            i, j = x
            st = j if st == i else i if st == j else st
        elif task == "T5":
            st[x] ^= 1
        else:
            i, j = x
            st[i], st[j] = st[j], st[i]
        trace.append(get(st))
    return trace, st


# --- rendering ----------------------------------------------------------------------------------


def render(task, lat, fmt):
    trace, full = run(task, lat)
    init, steps, q = lat["init"], lat["steps"], lat["q"]
    if task == "T1":
        cands = [str(i) for i in range(10)]
        ans = str(trace[-1])
        if fmt == "nl":
            prefix = ("Keep track of four variables. At the start, "
                      + ", ".join(f"{v} = {init[v]}" for v in VARS) + ".\nThen these updates happen in order:\n")
            body = "".join(f"Set {v} to {x}.\n" for v, x in steps)
            question = f"What is the value of {q} at the end? Reply with just the number."
        else:
            prefix = "Variables: " + " ".join(f"{v}={init[v]}" for v in VARS) + "\nUpdates:"
            body = "".join(f" {v}={x}" for v, x in steps) + "\n"
            question = f"Final {q}? Answer with one digit."
    elif task == "T2":
        if fmt == "nl":
            prefix = (f"A light is {'on' if init else 'off'}. People walk past its switch one at a time; "
                      "each person either flips the switch or leaves it alone.\n")
            body = "".join(f"{nm} {'flips the switch' if f else 'leaves it alone'}.\n"
                           for nm, f in zip(lat["names"], steps))
            question = "Is the light on or off now? Reply with just on or off."
            ans, cands = ("on" if trace[-1] else "off"), ["on", "off"]
        else:
            prefix = f"Bit starts at {init}. Ops (T = toggle, N = no-op):"
            body = "".join(" T" if f else " N" for f in steps) + "\n"
            question = "Final bit? Answer 0 or 1."
            ans, cands = str(trace[-1]), ["0", "1"]
    elif task == "T3":
        ans, cands = CUPS[trace[-1]], list(CUPS)
        if fmt == "nl":
            prefix = (f"Three cups, A, B and C, stand upside down on a table. A ball is under cup "
                      f"{CUPS[init]}. The cups are then swapped around, one pair at a time:\n")
            body = "".join(f"Swap cups {CUPS[i]} and {CUPS[j]}.\n" for i, j in steps)
            question = "Which cup is the ball under now? Reply with just the letter."
        else:
            prefix = f"Ball under {CUPS[init]}. Swaps:"
            body = "".join(f" {CUPS[i]}{CUPS[j]}" for i, j in steps) + "\n"
            question = "Ball under? Answer A, B or C."
    elif task == "T5":
        if fmt == "nl":
            prefix = ("Four lights, a, b, c and d, each have their own switch. At the start "
                      + ", ".join(f"light {s} is {'on' if init[i] else 'off'}" for i, s in enumerate(SWITCHES))
                      + ". People walk past one at a time; each person flips exactly one switch.\n")
            body = "".join(f"{nm} flips switch {SWITCHES[i]}.\n" for nm, i in zip(lat["names"], steps))
            question = f"Is light {SWITCHES[q]} on or off now? Reply with just on or off."
            ans, cands = ("on" if trace[-1] else "off"), ["on", "off"]
        else:
            prefix = ("Switches: " + " ".join(f"{s}={init[i]}" for i, s in enumerate(SWITCHES))
                      + ". Each op toggles the named switch:")
            body = "".join(f" {SWITCHES[i]}" for i in steps) + "\n"
            question = f"Final {SWITCHES[q]}? Answer 0 or 1."
            ans, cands = str(trace[-1]), ["0", "1"]
    else:
        names = ITEMS if fmt == "nl" else SYM_ITEMS
        ans, cands = names[trace[-1]], list(names)
        if fmt == "nl":
            prefix = ("Five boxes in a row are numbered 1 to 5. At the start they hold: "
                      + ", ".join(f"box {b + 1} the {ITEMS[init[b]]}" for b in range(5))
                      + ". Then the contents of two boxes are swapped, again and again:\n")
            body = "".join(f"Swap boxes {i + 1} and {j + 1}.\n" for i, j in steps)
            question = f"What is in box {q + 1} now? Reply with just the item."
        else:
            prefix = "Boxes: " + " ".join(f"{b + 1}{SYM_ITEMS[init[b]]}" for b in range(5)) + "\nSwaps:"
            body = "".join(f" {i + 1}{j + 1}" for i, j in steps) + "\n"
            question = f"Box {q + 1}? Answer one letter a-e."
    if isinstance(full, dict):
        full = [full[v] for v in VARS]
    return dict(prefix=prefix, steps=body, question=question, answer=ans, cands=cands,
                trace=trace, full=full if isinstance(full, list) else [full])


SAMPLE = {"T1": s1, "T2": s2, "T3": s3, "T4": s4, "T5": s5}


def perturb(task, lat, rng):
    """A twin with one initial value changed so the final answer differs."""
    b = copy.deepcopy(lat)
    want = run(task, lat)[0][-1]
    for _ in range(100):
        if task == "T1":
            # change q's last-written value (the initial value if q is never written)
            ks = [k for k, (v, _) in enumerate(b["steps"]) if v == lat["q"]]
            if ks:
                v, x = b["steps"][ks[-1]]
                b["steps"][ks[-1]] = (v, rng.choice([y for y in range(10) if y != x]))
            else:
                b["init"][lat["q"]] = rng.choice([y for y in range(10) if y != lat["init"][lat["q"]]])
        elif task == "T2":
            b["init"] = 1 - lat["init"]
        elif task == "T3":
            b["init"] = rng.choice([c for c in range(3) if c != lat["init"]])
        elif task == "T5":  # flip the queried switch's initial state (steps unchanged)
            b["init"] = list(lat["init"])
            b["init"][lat["q"]] ^= 1
        else:
            i, j = rng.sample(range(5), 2)
            b["init"] = list(lat["init"])
            b["init"][i], b["init"][j] = b["init"][j], b["init"][i]
        if run(task, b)[0][-1] != want:
            return b
        b = copy.deepcopy(lat)
    raise RuntimeError("no twin")


def _item(task, lat, fmt, n, idx, filler):
    r = render(task, lat, fmt)
    return dict(task=task, fmt=fmt, n=n, idx=idx, filler=filler, fill=filler_text(filler), **r)


def make(task: str, n_steps: int, fmt: str, k: int, seed: int = 0, filler: int = 0) -> list[dict]:
    rng = random.Random(f"{task}-{fmt}-{n_steps}-{seed}")
    return [_item(task, SAMPLE[task](rng, n_steps), fmt, n_steps, i, filler) for i in range(k)]


def make_pairs(task, n_steps, fmt, k, seed=0, filler=0):
    """k (A, B) twins: same steps, one initial value changed, different answers.

    T1 changes the queried variable's last written value instead (in the steps when it is
    written at all), so its twins differ inside the steps."""
    rng = random.Random(f"pair-{task}-{fmt}-{n_steps}-{seed}")
    out = []
    for i in range(k):
        a = SAMPLE[task](rng, n_steps)
        b = perturb(task, a, rng)
        out.append((_item(task, a, fmt, n_steps, i, filler), _item(task, b, fmt, n_steps, i, filler)))
    return out


def user_text(it: dict) -> str:
    return it["prefix"] + it["steps"] + it["fill"] + it["question"]


def check(it: dict) -> None:
    """Re-derive the answer from the rendered text (guards against generator bugs)."""
    t, s = it["task"], it["steps"]
    if t == "T1":
        q = re.search(r"(?:value of|Final) (\w)", it["question"]).group(1)
        init = dict(re.findall(r"(\w) ?= ?(\d)", it["prefix"]))
        upd = re.findall(r"Set (\w) to (\d)", s) or re.findall(r"(\w)=(\d)", s)
        val = init[q]
        for v, x in upd:
            if v == q:
                val = x
        assert val == it["answer"], (val, it["answer"])
    elif t == "T2":
        if it["fmt"] == "nl":
            on = (" is on." in it["prefix"]) ^ (s.count("flips") % 2 == 1)
            assert ("on" if on else "off") == it["answer"]
        else:
            b = int(re.search(r"starts at (\d)", it["prefix"]).group(1)) ^ (s.count("T") % 2)
            assert str(b) == it["answer"]
    elif t == "T3":
        ball = re.search(r"under (?:cup )?(\w)\.", it["prefix"]).group(1)
        pairs = re.findall(r"Swap cups (\w) and (\w)", s) or re.findall(r"([ABC])([ABC])", s)
        for i, j in pairs:
            ball = j if ball == i else i if ball == j else ball
        assert ball == it["answer"]
    elif t == "T5":
        if it["fmt"] == "nl":
            st = {k: v == "on" for k, v in re.findall(r"light (\w) is (on|off)", it["prefix"])}
            flips = re.findall(r"flips switch (\w)", s)
            q = re.search(r"Is light (\w)", it["question"]).group(1)
        else:
            st = {k: v == "1" for k, v in re.findall(r"(\w)=(\d)", it["prefix"])}
            flips = re.findall(r" ([a-d])", s)
            q = re.search(r"Final (\w)\?", it["question"]).group(1)
        assert len(flips) == it["n"]
        for k in flips:
            st[k] = not st[k]
        want = ("on" if st[q] else "off") if it["fmt"] == "nl" else str(int(st[q]))
        assert want == it["answer"], (want, it["answer"])
    elif t == "T4":
        if it["fmt"] == "nl":
            boxes = {int(b): x for b, x in re.findall(r"box (\d) the (\w+)", it["prefix"])}
            pairs = re.findall(r"Swap boxes (\d) and (\d)", s)
            q = int(re.search(r"box (\d) now", it["question"]).group(1))
        else:
            boxes = {int(b): x for b, x in re.findall(r"(\d)([a-e])", it["prefix"])}
            pairs = re.findall(r"(\d)(\d)", s)
            q = int(re.search(r"Box (\d)", it["question"]).group(1))
        for i, j in pairs:
            i, j = int(i), int(j)
            boxes[i], boxes[j] = boxes[j], boxes[i]
        assert boxes[q] == it["answer"]


if __name__ == "__main__":
    from collections import Counter

    for t in TASKS + EXTRA_TASKS:
        for f in FORMATS:
            for n in LENGTHS:
                items = make(t, n, f, 200)
                for it in items:
                    check(it)
                if n == 64:
                    print(t, f, "answers", sorted(Counter(i["answer"] for i in items).items()))
            for a, b in make_pairs(t, 16, f, 50):
                check(a), check(b)
                assert a["answer"] != b["answer"] and a["question"] == b["question"]
                if t != "T1":
                    assert a["steps"] == b["steps"]
            ex = make(t, 4, f, 1)[0]
            print(user_text(ex), "->", ex["answer"], ex["full"], "\n")

"""Reproduces the out-of-order-ASR paragraph scrambling, and shows it fixed.

v7 numbered paragraphs by position, so a late-arriving chunk renumbered every
paragraph after it. The client upserts by paragraph_id, so the text landed in
the wrong slot and stale content stayed on screen.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.session.transcript import TranscriptStore


# ---------------------------------------------------------------- v7 baseline
def v7_paragraphs(chunks):
    """Faithful reimplementation of v7 TranscriptStore.paragraphs()."""
    paras = []
    for c in sorted(chunks, key=lambda x: (x["start"], x["seg"])):
        spk = f"Speaker {c['label']}"
        if paras and paras[-1]["speaker"] == spk:
            paras[-1]["text"] += " " + c["text"]
            paras[-1]["end"] = c["end"]
        else:
            paras.append({
                "paragraph_id": len(paras) + 1,     # POSITIONAL — the bug
                "speaker": spk, "text": c["text"],
                "start": c["start"], "end": c["end"],
            })
    return paras


def v7_add(chunks, new):
    """Faithful v7 TranscriptStore.add(): returns the ONE paragraph containing
    the new chunk. main.py then did `await ws.send_json(paragraph)` — a single
    paragraph per chunk, never the whole list."""
    chunks.append(new)
    chunks.sort(key=lambda x: (x["start"], x["seg"]))
    paras = v7_paragraphs(chunks)
    pos = 0
    for i, p in enumerate(paras):
        if p["start"] <= new["start"] <= p["end"]:
            pos = i
    return paras[pos]


def client_upsert(state, paras):
    """What TranscriptView.useTranscript() does with a `transcript` message."""
    for p in paras:
        state[p["paragraph_id"]] = p
    return state


def test_v7_scrambles_on_out_of_order_arrival():
    """A-B-A speaker pattern, with B's ASR completing last.

    Perfectly plausible with ASR_CONCURRENCY=6: B was the longest chunk, or
    just got a slower Gemini response. Nothing about the audio is unusual.
    """
    A = {"start": 0.0, "end": 2.0, "seg": 1, "label": 1, "text": "hello"}
    B = {"start": 2.5, "end": 4.5, "seg": 2, "label": 2, "text": "second"}
    C = {"start": 5.0, "end": 7.0, "seg": 3, "label": 1, "text": "third"}

    chunks, screen = [], {}
    for new in (A, C, B):                       # B lands last
        screen = client_upsert(screen, [v7_add(chunks, new)])

    rendered = [screen[k]["text"] for k in sorted(screen)]
    truth = ["hello", "second", "third"]
    print(f"    v7 on screen : {rendered}")
    print(f"    ground truth : {truth}")
    assert rendered != truth, "expected v7 to scramble"
    print("  v7: late chunk renumbered paragraphs, screen is wrong  (bug reproduced)")


def test_v10_survives_out_of_order_arrival():
    store = TranscriptStore()
    # Chunk ids are reserved in AUDIO order, before ASR is called.
    c1 = store.new_chunk(seg_id=1, start=0.0, end=2.0)
    c2 = store.new_chunk(seg_id=2, start=2.5, end=4.5)
    c3 = store.new_chunk(seg_id=3, start=5.0, end=7.0)
    for c, s in ((c1, 0), (c2, 1), (c3, 0)):
        c.speaker = s
        c.language = "si"

    screen = {}
    # ASR completes out of order: 1, then 3, then 2.
    for c, txt in ((c1, "hello"), (c3, "third"), (c2, "second")):
        c.text = txt
        paras, mode = store.diff()
        if mode == "refresh":
            screen = {p["paragraph_id"]: p for p in paras}   # REPLACE
        elif mode == "append":
            screen = client_upsert(screen, paras)

    rendered = [screen[k]["text"] for k in sorted(screen, key=lambda k: screen[k]["start"])]
    print(f"    v10 on screen: {rendered}")
    assert rendered == ["hello", "second", "third"], rendered
    print("  v10: stable ids + refresh-on-restructure, screen is correct  OK")


def test_relabel_merges_adjacent_paragraphs():
    """When a re-cluster decides two 'speakers' were one person, the two
    paragraphs must MERGE — the case that forces a refresh rather than an
    upsert."""
    store = TranscriptStore()
    for i, (a, b, txt) in enumerate(
        [(0.0, 2.0, "one"), (2.5, 4.5, "two"), (5.0, 7.0, "three")], start=1
    ):
        c = store.new_chunk(seg_id=i, start=a, end=b)
        c.text, c.language = txt, "en"
        c.speaker = 0 if i != 2 else 1     # phantom speaker in the middle

    assert len(store.paragraphs()) == 3
    store.diff()

    # A later pass corrects the phantom away.
    store.relabel(lambda s, e: 0)
    paras, mode = store.diff()
    assert mode == "refresh", mode
    assert len(paras) == 1, paras
    assert paras[0]["text"] == "one two three"
    print("  relabel collapsing a phantom: 3 paragraphs -> 1, mode=refresh  OK")


def test_code_switched_paragraph_reports_the_mix():
    store = TranscriptStore()
    for i, (a, b, txt, lang) in enumerate(
        [(0.0, 2.0, "ayubowan", "si"), (2.2, 4.0, "how are you", "en")], start=1
    ):
        c = store.new_chunk(seg_id=i, start=a, end=b)
        c.text, c.language, c.speaker = txt, lang, 0
    p = store.paragraphs()[0]
    assert p["language"] == "si+en", p["language"]
    print(f"  code-switched paragraph reports language='{p['language']}'  OK")


if __name__ == "__main__":
    print("TranscriptStore tests\n")
    test_v7_scrambles_on_out_of_order_arrival()
    test_v10_survives_out_of_order_arrival()
    test_relabel_merges_adjacent_paragraphs()
    test_code_switched_paragraph_reports_the_mix()
    print("\nall passed")

import json
d = json.load(open(r"C:\projects\local\audio_processer\work\diar\final_diar_classification.json", encoding="utf-8"))
print({k: v for k, v in d.items() if k != "clusters"})
cl = [c for c in d["clusters"] if c["label"] != "nonspeech"]
cl.sort(key=lambda c: -c["union_s"])
print("speech clusters:", len(cl))
for c in cl[:15]:
    txt = (c.get("asr_text") or "")[:70].replace("\n", " ")
    print(f"spk{c['speaker']:>3} dur={c['union_s']:8.2f}s nseg={c['n_segments']:>4} cos={c['cos_to_main']:.3f} "
          f"label={c['label']:>6} chars_per_s={c['chars_per_s']:.2f} | {txt}")

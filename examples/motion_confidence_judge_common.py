# SPDX-License-Identifier: Apache-2.0
"""Shared judge machinery for the independent-VLM-judge QA scripts
(``motion_confidence_negative_qa.py`` and ``motion_confidence_self_coc_qa.py``):
scene-image construction, chat-message building, CoT-then-verdict prompting,
verdict parsing, and majority-vote-over-N-samples aggregation.

Reliability improvements (see ``docs/motion_confidence_experiment.md`` Experiment 7 --
smoke test surfaced judge-noise on near-identical phrasings, split verdicts on
essentially the same claim):

1. CoT-before-answer prompting: the judge is asked to describe the scene and reason
   step by step BEFORE committing to a verdict, ending with a final line containing
   exactly one word (Yes/No/Uncertain), instead of the old verdict-first format.
2. Majority vote over 3 independent samples (``do_sample=True``, moderate temperature)
   instead of a single greedy call, with all 3 raw verdicts+explanations recorded
   alongside the final majority verdict so disagreement can be inspected later.
"""

import re

CAMERA_LABELS = [
    "Front left camera",
    "Front camera",
    "Front right camera",
    "Rear left camera",
    "Rear camera",
    "Rear right camera",
]

# Legacy verdict-first prompt (kept for reference / A-B comparisons); superseded by
# JUDGE_PROMPT_COT below as the default.
JUDGE_PROMPT_VERDICT_FIRST = (
    "Given this driving scene, is the following a reasonable and correct driving "
    "decision for the ego vehicle right now: '{action}'? Answer with exactly one word "
    "first -- Yes, No, or Uncertain -- then a brief one-sentence explanation."
)

JUDGE_PROMPT_COT = (
    "Given this driving scene, carefully describe what's relevant in the scene, then "
    "reason step by step about whether the following is a reasonable and correct "
    "driving decision for the ego vehicle right now: '{action}'. After your reasoning, "
    "end with a final line containing exactly one word: Yes, No, or Uncertain."
)

SYSTEM_PROMPT = (
    "You are an expert driving-scene assessor. You will be shown the six camera views "
    "(front-left, front, front-right, rear-left, rear, rear-right) surrounding an ego "
    "vehicle at one moment in time. Judge only whether the proposed driving decision "
    "fits the scene shown."
)

_VERDICT_WORDS = ("yes", "no", "uncertain")
_FINAL_LABEL_RE = re.compile(
    r"(?:final\s*(?:answer|verdict)|verdict|answer)\s*[:\-]\s*\**\s*(yes|no|uncertain)\b",
    re.IGNORECASE,
)


def build_pil_images(source_data, vqa_data, current_frame_idx=3):
    """image_frames: (N_cameras=6, num_frames=4, 3, H, W) uint8, oldest..newest. Take
    only the current (most recent, t0) frame per camera -- the judge question is about
    "right now", and 6 single-timestep images keep the per-call cost manageable."""
    from PIL import Image

    image_frames = vqa_data["image_frames"]
    n_frames = image_frames.shape[1]
    idx = min(current_frame_idx, n_frames - 1)
    frames = image_frames[:, idx]  # (6, 3, H, W) uint8
    pil_images = []
    for cam_i in range(frames.shape[0]):
        arr = frames[cam_i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) uint8
        pil_images.append(Image.fromarray(arr))
    return pil_images


def build_messages(pil_images, action, prompt_template=JUDGE_PROMPT_COT):
    content = []
    for label, img in zip(CAMERA_LABELS, pil_images):
        content.append({"type": "text", "text": f"{label}:"})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt_template.format(action=action)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_verdict(text):
    """Legacy verdict-first parsing: first word of the response -> Yes/No/Uncertain,
    else 'unparseable'. Kept for the old verdict-first prompt."""
    stripped = text.strip()
    if not stripped:
        return "unparseable"
    first = stripped.split()[0].strip(".,:;!'\"()").lower()
    if first in _VERDICT_WORDS:
        return first.capitalize()
    return "unparseable"


def parse_verdict_cot(text):
    """Verdict parsing for the CoT-then-answer prompt: the verdict is expected on its
    own final line, but the model doesn't always follow that exactly, so we fall back
    progressively:

    1. Scan non-empty lines from the end; return the first line that, after stripping
       markdown emphasis/punctuation, is exactly "yes"/"no"/"uncertain".
    2. Scan lines from the end for a labelled verdict like "Final answer: Yes" or
       "Verdict: No".
    3. 'unparseable' if neither of the above match -- this deliberately does NOT fall
       back to "the last occurrence of yes/no/uncertain anywhere in the text": reasoning
       prose is full of incidental negations ("no indication that...", "no immediate
       hazard...") that are not verdicts, and a truncated response (generation hit
       max_new_tokens before reaching its final line) will very often end mid-sentence
       on exactly one of these -- scanning the whole text for the last bare occurrence
       of "no" reliably mis-reads a truncated, Yes-trending explanation as a "No"
       verdict. An unparsed/truncated sample should be reported as 'unparseable'
       (and can pull the majority vote to 'no_majority'), not silently guessed.
    """
    stripped = text.strip()
    if not stripped:
        return "unparseable"

    lines = [l.strip() for l in stripped.splitlines() if l.strip()]

    for line in reversed(lines):
        cleaned = line.strip("*_#> \t").strip(".,:;!'\"()").strip()
        if cleaned.lower() in _VERDICT_WORDS:
            return cleaned.lower().capitalize()

    for line in reversed(lines):
        m = _FINAL_LABEL_RE.search(line)
        if m:
            return m.group(1).lower().capitalize()

    return "unparseable"


def majority_vote(verdicts):
    """2-of-3 majority over a list of verdict strings. If all entries are pairwise
    distinct (no value appears >= 2 times), returns 'no_majority' rather than guessing."""
    from collections import Counter

    counts = Counter(verdicts)
    top_value, top_count = counts.most_common(1)[0]
    if top_count >= 2:
        return top_value
    return "no_majority"


def generate_judge_samples(
    model,
    processor,
    process_vision_info,
    device,
    pil_images,
    action,
    prompt_template=JUDGE_PROMPT_COT,
    max_new_tokens=384,
    num_samples=3,
    temperature=0.7,
    top_p=0.95,
):
    """Run the judge ``num_samples`` times (independent samples, do_sample=True) on one
    candidate action, in a single batched ``generate`` call via ``num_return_sequences``.
    Returns a list of ``{"verdict": ..., "explanation": ...}`` dicts, one per sample, plus
    the aggregated majority verdict as a 2-tuple: ``(samples, majority_verdict)``.
    """
    import torch

    messages = build_messages(pil_images, action, prompt_template)
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
        )
    trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
    out_texts = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    samples = []
    for out_text in out_texts:
        verdict = parse_verdict_cot(out_text)
        samples.append({"verdict": verdict, "explanation": out_text.strip()})

    majority_verdict = majority_vote([s["verdict"] for s in samples])
    return samples, majority_verdict

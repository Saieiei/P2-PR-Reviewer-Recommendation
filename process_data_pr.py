#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_data_pr.py — build train/dev/test TSVs from preprocessed_data.json
 • no more early‐skipping of records with zero positives
 • progress bars for both the record loop and negative‐sampling
"""

import json
import random
import re
import os
from typing import List, Tuple
from difflib import SequenceMatcher
from tqdm import tqdm
from rapidfuzz import fuzz

# === Configuration ===
DATA_FILE       = "preprocessed_data.json"
OUTPUT_DIR      = "."              # where train.tsv, dev.tsv, test.tsv go
CODESPLIT       = "<CODESPLIT>"    # separator for run_classifier.py
TRAIN_RATIO     = 0.80
DEV_RATIO       = 0.10
SEED            = 42

# Augmentation settings
WINDOW_SIZE     = 200  # tokens per window
OVERLAP         = 50   # tokens overlap between windows
SUGG_OVERSAMPLE = 2    # how many times to duplicate suggestion positives

# -------------------------------------------------------------------
# 1)  Trivial‐comment detection (skip these if no suggestion)
# -------------------------------------------------------------------
_TRIVIAL_PATTERNS = [
    r"^\s*(?:lgtm|ship it|\+1)[.!]*$",
    r"^\s*(?:thanks?|thank you)[.!]*$",
]
_TRIVIAL_RE = [re.compile(p, re.I) for p in _TRIVIAL_PATTERNS]

def is_trivial(text: str) -> bool:
    txt = (text or "").strip()
    # too short or no alphanumerics
    if len(txt) < 5 or not re.search(r"[A-Za-z0-9]", txt):
        return True
    # matches trivial patterns
    for pat in _TRIVIAL_RE:
        if pat.match(txt):
            return True
    return False

# -------------------------------------------------------------------
# 2)  Hand-picked generic comment examples (skip these even if >5 chars)
# -------------------------------------------------------------------
GENERIC_EXAMPLES = [
    "Applied.",
    "Sorry!",
    "Yeah that is probably better. Thanks. I updated the patch.",
    "Use poison",
    "takeName",
    "Side hack, probably shouldn't do this here",
    "```suggestion```Noise",
    "Fixed",
    "I'll post that tomorrow in the hope that the build bots have calmed down by then, since precommit seems to be failing for everyone atm",
    "I think there are two things going on here:",
    "Nit: `m_`?",
    "The main changes are here.",
    "This is the type of typo this can help prevent!",
    "Nice!",
    "Seems like this variant is unnecessary.",
    "nit: What is this needed for?",
    "Delete this line as it's irrelevant to this patch?",
    "Delete both lines?",
    "Thank you for the suggestion; I’ll take it into consideration.",
    "Braces",
    "also here",
    "Didn't see, thank you",
    "Thanks. I think it's fine either way.",
    "This should have a different text.",
    "I've added some tests with different attributes in different places. If you have any idea what should be added - small example (just code line) would be appreciated.",
    "Thanks, fixed. Now it mention that empty function also merged.",
    "Thanks, fixed",
    "fixed",
    "Pushed a correction. It still passes here locally.",
    "Thank you.",
    "Should we also send notification on Discord?",
    "Maybe on discord, just send a link to the discourse thread? IMHO, it is better to only have to manage one thread.",
    "```suggestion\n}\n\n```",
    "This is kind of worse",
    "whether",
    "Done. Thanks for the finding.",
    "Thank you for the link!",
    "Thank you for the link!\n\nDone.",
    "Agree. Early return is a good idea here.",
    "Done.\n\nIt looks better now.\n\nThank you!",
    "Good idea, done.",
    "Comment",
    "Correct, I forgot to add one. Will add.",
    "`false`?",
    "Ditto.",
    "yes, thanks for the notice",
    "typo: \"sanboxir\"",
    "Fixed.",
    "I've added those tests now.",
    "Ah! I've removed the escape. :-)",
    "Unrelated?",
    "Ok, makes sense. Just pushed something.",
    "ok, done!",
    "Done.",
    "This was the primary goal of this PR.",
    "Oops yes, removed",
    "Okay, I see. I'll amend it.",
    "This does not look obvious to me. Can you explain?",
    "I fixed that / added comments",
    "Can we add some comments? what causes us to change it from 24 -> 32",
    "another typo :)",
    "Thanks for the finding. Done",
    "Thanks for the hint. That makes more sense. Feel free to check the corresponding commits.",
    "Thank you for the explanations. Am I right in assuming that we can leave it as it is for this PR?",
    "I think it's good as-is. Thanks.",
    "oh yes, thanks",
    "fixed in main, thanks",
    "Sounds good! Worth adding a comment explaining that so others don't wonder as well.",
    "Actually, I don't see it anymore.  Good then.",
    "Sounds good! That seems to also be how we do it elsewhere in the codebase",
    "Nit: Update comment above.",
    "Yes, this seems clear.",
    "Updated",
    "Changed to suggestion",
    "Added comment",
    "Added test for upper limit",
    "OK, I'll take into account when implementing next intrinsics.",
    "I tried that fix, and I agree it looks like the right fix to me. It seems to solve the issue.",
    "https://github.com/llvm/llvm-project/pull/133840",
    "```suggestion\n```",
    "fixed.",
    "Ahh, got it.",
    "yep, changing it now",
    "This looks to be resolve, but not done.",
    "Thanks a lot for the comment!",
    "Done.",
    "done!",
    "also here",
    "(see comment above)",
    "Yes, that will work.",
    "Not really relevant here, and redundant with the final result check",
    "Yea, updated.",
    "same here",
    "You're right. I created https://github.com/llvm/llvm-project/pull/133370",
    "What's all this about?",
    "Oops needed to remove that",
    "Nit: newline",
    "moved it back",
    "remove?",
    "hahah, epic explanation!",
    "Removed",
    "This is already tested in mlir/test/Dialect/LLVMIR/func.mlir",
    "I see. This makes sense. Thanks for clarifying.",
    "unrelated?",
    "unnecessary",
    "remove",
    "unrelated",
    "Done.",
    "Yes, improved. Added more check lines to show that.",
    "Sorry, I'm too spoiled with Python 3.12 on my job :-)",
    "Same here.",
    "Good catch! Will add tomorrow.",
    "ahh, didn't read the full code.",
    "That's great to know!",
    "What's this for?",
    "Good call, that exposed an issue. I'll add test coverage and correct the problem.",
    "Absolutely!",
    "Fair point!",
    "I'm blind!",
    "Sure, added!",
    "Sure ??",
    "You are right. Fixed in the latest revision.",
    "done, but what's the difference?",
    "or false, whichever.",
    "Ok, yeah that makes sense.",
    "Makes sense, thanks for explaining!",
    "This still worries me.",
    "What do you expect here?",
    "Ok, will do actual cost estimation",
    "Done.",
    "Ok. Thanks for checking.",
    "ditto",
    "Thanks ! I didn't know they could be implicit.",
    "Nope. :-(\nBut now fixed.",
    "correct",
    "This also unintentionally?",
    "Is this left unintentionally?",
    "Good idea.",
    "Removed.",
    "I trimmed most of the comments.",
    "Thanks, will do.",
    "Ohh I see what you mean now. Thanks",
    "I think this could use a comment",
    "Why remove the comment?",
    "(I mailed core)",
    "@zygoloid",
    "Here too perhaps?",
    "They're not, I'll add this as a test case.",
    "I'll add test coverage",
    "Ah, yeah, makes sense",
    "Was this a previous bug?",
    "I have to say that is a choice.",
    "Thank you for the feedback. Will do.",
    "Sure, good call!",
    "Surely you mean 21?",
    "Thank you, I wasn't sure how this work related to releases. I fixed it.",
    "thank you for the review. I have removed extra line.",
    "Thank you for the suggestion! I've added a [comment ](https://github.com/llvm/llvm-project/issues/128656#issuecomment-2753046448) in the issue #128656 to track these TODO tasks.",
    "You are right! Correcting now.",
    "this as well",
    "yes. will do",
    "will remove it",
    "Oh, that's actually not used, I should get rid of it thanks!",
    "https://github.com/llvm/llvm-project/pull/133633",
    "Also here",
    "I agree, I'll revert these changes.",
    "I have kept it as-is if you don't mind.",
    "And this key is new.",
    "I don't think this is necessary.",
    "Sure, I'll do that.",
    "add negative test",
    "Is this supposed to be on 2 lines?",
    "good catch. no this should be one line",
    "Good points, thank you! I've updated the PR accordingly.",
    "That sounds great to me, I'd support that",
    "Fixed in latest diff.",
    "newline :)",
    "Is this being used somewhere?",
    "Fixed this.",
    "Fixed that.",
    "Yup, fixed that.",
    "No, idk how that got added. Deleted it.",
    "Fixed.",
    "Fair enough, I reverted this file.",
    "And this key is new.",
    "I don't think this is necessary.",
    "Sure, I'll do that.",
    "add negative test",
    "good catch. no this should be one line",
    "Good points, thank you! I've updated the PR accordingly.",
    "Fixed in latest diff.",
    "Thanks, fixed",
    "Changes applied Let me know if it looks good now",
    "Sure.",
    "This has been fixed.",
    "Done.  This preserves consistency.",
    "Ok. I can update it this way.",
    "Added",
    "Fixed, thanks!",
    "removed!",
    "Fixed, thanks!",
    "Here and below.",
    "Thanks, should be addressed now",
    "Okay feel free to ignore my comment then",
    "It shouldn't.",
    "Ah, right",
    "Code updated",
    "This is worse",
    "No problem.",
    "Please ignore.",
    "What is this for?",
    "Skipped for now: https://github.com/llvm/llvm-project/commit/d4002b43f517fea0292bf71dccaa3d0f6dd798b9",
    "parameters",
    "Woops, removed!",
    "Good point, removed",
    "You're right, sorry. I've fixed this now.",
    "Thanks, that makes sense. I've renamed this.",
    "Done, apologies for the noise.",
    "This needs to be updated",
    "Same thing with header here",
    "@Moxinilian ??",
    "I think i have something cleaner let me know in the new pr.",
    "Ah yeah, I read that wrong. Thanks for confirming.",
    "That's a great point and I really appreciate the suggestion. In fact, I wasn't sure how to do it : ) That's now been incorporated.",
    "Yeah that is probably better. Thanks. I updated the patch.",
    "Yeah. That's ok",
    "Awesome. Thank you!",
    "I never noticed this. Thanks!"
]

# 2) pick your fuzzy cutoff
FUZZY_THRESHOLD = 90

# 3) normalize to lowercase alpha-numeric “tokens”
def normalize(s: str) -> str:
    # replace any run of non-word chars with a single space
    return re.sub(r"\W+", " ", s).strip().lower()

# 4) pre-normalize your generic set once
"""
  Return True if `comment` fuzzily matches any of your generic examples
  (ignores word order, extra punctuation, small typos).
"""
GENERIC_NORMS = [normalize(g) for g in GENERIC_EXAMPLES]

def is_generic(comment: str) -> bool:
    c = normalize(comment)
    for g in GENERIC_NORMS:
        # token_set_ratio ignores word order & duplicate tokens
        if fuzz.token_set_ratio(c, g) >= FUZZY_THRESHOLD:
            return True
    return False

# -------------------------------------------------------------------
# 3)  Sliding‐window for long diffs
# -------------------------------------------------------------------
def sliding_windows(diff: str) -> List[str]:
    tokens = diff.split()
    if len(tokens) <= WINDOW_SIZE:
        return [" ".join(tokens)]
    windows = []
    step = WINDOW_SIZE - OVERLAP
    for start in range(0, len(tokens), step):
        windows.append(" ".join(tokens[start : start + WINDOW_SIZE]))
        if start + WINDOW_SIZE >= len(tokens):
            break
    return windows

# -------------------------------------------------------------------
# 4)  Build positives & negatives, split, write TSVs
# -------------------------------------------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

positives: List[Tuple[int,str,str,str,str]] = []

# wrap main loop in tqdm so you see progress
for rec in tqdm(records, desc="Scanning JSON records"):
    pr        = str(rec.get("pr_number",""))
    fn        = rec.get("filename","")
    labels    = rec.get("labels","")
    diff_txt  = (rec.get("diff_text","") or "").strip()
    sugg      = (rec.get("suggestion_text","") or "").strip()
    comm      = (rec.get("comment_text","") or "").strip()
    commenter = rec.get("commenter","")

    # still drop records with absolutely no diff context (no point in training on them)
    if not diff_txt:
        continue

    prefix = f"[FILE]{fn} [LABEL]{labels} "
    for win in sliding_windows(diff_txt):
        diff_win = prefix + win

        # (1) Suggestions → always positives (oversample)
        if sugg:
            for _ in range(SUGG_OVERSAMPLE):
                positives.append((1, pr, commenter, diff_win, sugg))

        # (2) Comments → positives only if non-trivial & not generic
        if (comm
            and comm != sugg
            and not is_trivial(comm)
            and not is_generic(comm)
        ):
            positives.append((1, pr, commenter, diff_win, comm))

print(f"> Created {len(positives)} positive examples")

# Negatives: one per positive, drawn from other PRs
random.seed(SEED)
negatives: List[Tuple[int,str,str,str,str]] = []
idxs = list(range(len(positives)))

# wrap negative sampling in tqdm too
for i in tqdm(idxs, desc="Sampling negatives"):
    _, pr_i, _, diff_i, _ = positives[i]
    while True:
        j = random.choice(idxs)
        if j != i and positives[j][1] != pr_i:
            break
    _, pr_j, commenter_j, _, text_j = positives[j]
    negatives.append((0, pr_i, commenter_j, diff_i, text_j))

print(f"> Created {len(negatives)} negative examples")

# Combine, shuffle, split
all_ex = positives + negatives
random.shuffle(all_ex)

n_total = len(all_ex)
n_train = int(n_total * TRAIN_RATIO)
n_dev   = int(n_total * DEV_RATIO)

train_set = all_ex[:n_train]
dev_set   = all_ex[n_train : n_train + n_dev]
test_set  = all_ex[n_train + n_dev :]

print(f"> Train/dev/test counts: {len(train_set)}/{len(dev_set)}/{len(test_set)}")

def write_tsv(exs: List[Tuple[int,str,str,str,str]], path: str):
    with open(path, "w", encoding="utf-8") as out:
        for label, prnum, commenter, diffa, commb in exs:
            d = diffa.replace("\n"," ")
            c = commb.replace("\n"," ")
            out.write(f"{label}{CODESPLIT}{prnum}{CODESPLIT}{commenter}{CODESPLIT}{d}{CODESPLIT}{c}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)
write_tsv(train_set, os.path.join(OUTPUT_DIR, "train.tsv"))
write_tsv(dev_set,   os.path.join(OUTPUT_DIR, "dev.tsv"))
write_tsv(test_set,  os.path.join(OUTPUT_DIR, "test.tsv"))

print("✔ process_data_pr.py complete — train/dev/test ready for run_classifier.py")

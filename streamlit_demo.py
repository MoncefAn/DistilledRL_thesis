# streamlit_demo.py
# Run: streamlit run streamlit_demo.py

import streamlit as st
import json
import re
import math
import os

from math_verify import parse as mv_parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig


# ============================================================
# PAGE CONFIG + CSS (consolidated — everything in one block)
# ============================================================

st.set_page_config(page_title="DistilledRL Demo", layout="wide", page_icon="🧪")
st.cache_data.clear()

st.markdown("""
<style>
.column-card {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    border: 1px solid #e0e0e0;
    border-top: 4px solid #d0d4d9;
}
.column-card.correct { border-top-color: #2ea043; }
.column-card.wrong   { border-top-color: #cf222e; }

.model-header { font-size: 18px; font-weight: 600; margin-bottom: 8px; }

.stat-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 4px;
    margin-bottom: 4px;
}
.badge-green { background: #d4edda; color: #155724; }
.badge-red   { background: #f8d7da; color: #721c24; }
.badge-grey  { background: #e9ecef; color: #495057; }

.reasoning {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    line-height: 1.55;
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 400px;
    overflow-y: auto;
    background: #ffffff;
    padding: 8px 12px;
    border-radius: 4px;
    border: 1px solid #e9ecef;
}
.sample-wrap.correct .reasoning { border-left: 3px solid #2ea043; }
.sample-wrap.wrong   .reasoning { border-left: 3px solid #cf222e; }

.gt-pill {
    background: #fff3cd;
    color: #856404;
    padding: 6px 16px;
    border-radius: 16px;
    font-weight: 700;
    font-size: 15px;
    display: inline-block;
    margin-left: 8px;
}

.question-card {
    background: #f6f8fa;
    padding: 16px 20px;
    border-radius: 8px;
    border: 1px solid #e1e4e8;
    margin-bottom: 20px;
}

.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #8c959f;
    text-transform: uppercase;
    margin-top: 10px;
    margin-bottom: 3px;
}

/* ---------- Confidence gate heatmap ---------- */
.heatmap-container {
    background: #ffffff;
    padding: 16px 18px;
    border-radius: 8px;
    border: 1px solid #d0d7de;
    line-height: 2.0;
    word-wrap: break-word;
    white-space: normal;
}
.heatmap-line {
    display: block;
    min-height: 1.9em;
    line-height: 1.9;
}
.tok {
    display: inline-block;
    padding: 2px 3px;
    margin: 0 0 1px 0;
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13.5px;
    white-space: pre;
    transition: outline 0.1s;
    cursor: help;
}
.tok:hover { outline: 1.5px solid #0969da; }
.heatmap-legend {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 12px;
    color: #57606a;
    margin: 6px 0 14px 0;
}
.heatmap-swatch {
    display: inline-block;
    width: 80px;
    height: 14px;
    border-radius: 3px;
    background: linear-gradient(to right,
        rgba(88,166,255,0.05),
        rgba(88,166,255,0.95));
}
.gate-explanation {
    background: #f6f8fa;
    padding: 12px 16px;
    border-radius: 6px;
    border-left: 4px solid #0969da;
    font-size: 13px;
    color: #57606a;
    margin-bottom: 16px;
}
.heatmap-question {
    background: #fff8c5;
    padding: 10px 14px;
    border-radius: 6px;
    border: 1px solid #e6d870;
    font-size: 13px;
    margin-bottom: 12px;
    color: #4d3800;
}
.heatmap-question code {
    background: #ffffff;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# JUDGE / EXTRACTION
# ============================================================

def parse_number(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".:;, ")
    frac = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if frac:
        n, d = float(frac.group(1)), float(frac.group(2))
        return n / d if d != 0 else None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def extract_boxed_last(text):
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def trim_at_first_boxed(text):
    if not text:
        return text
    idx = text.find("\\boxed")
    if idx == -1:
        return text
    i = text.find("{", idx)
    if i == -1:
        return text
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[:j + 1]
    return text


ANSWER_PATTERNS = [
    r"(?:final\s+answer)\s*(?:is|=|:)\s*([^\n.]+)",
    r"(?:the\s+answer)\s*(?:is|=|:)\s*([^\n.]+)",
    r"(?:therefore|thus|so)\s*,?\s*(?:the\s+)?answer\s*(?:is|=|:)\s*([^\n.]+)",
    r"(?:has|have)\s+\$?(-?\d[\d,]*\.?\d*)\s*(?:dollars|left|remaining|\.|,|\s|$)",
]


def extract_answer_regex(text):
    if not text:
        return None

    # 1. boxed
    boxed = extract_boxed_last(text)
    if boxed is not None:
        v = parse_number(boxed)
        if v is not None:
            return v

    # 2. explicit patterns ("Final Answer: X", "the answer is X", "has X left")
    for pat in ANSWER_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            v = parse_number(matches[-1])
            if v is not None:
                return v

    # 3. Conclusion sentences (scan from the end)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in reversed(sentences):
        if not re.match(
            r"\s*(therefore|thus|so|hence|in total|total|finally|the answer)\b",
            sentence, re.I,
        ):
            continue

        # 3a. number immediately after "is / are / = / : / equals"
        #     Catches: "is $7,400", "= 694", ": 320", "equals 42"
        m = re.search(
            r"(?:is|are|was|were|equals?|=|:)\s*\$?\s*(-?\d[\d,]*\.?\d*)",
            sentence, re.I,
        )
        if m:
            v = parse_number(m.group(1))
            if v is not None:
                return v

        # 3b. last number NOT followed by a hyphen
        #     Skips "5-day", "3-dozen" style modifiers.
        nums = []
        for mm in re.finditer(r"-?\d[\d,]*\.?\d*", sentence):
            end = mm.end()
            if end < len(sentence) and sentence[end] == "-":
                continue
            nums.append(mm.group())
        if nums:
            v = parse_number(nums[-1])
            if v is not None:
                return v

    # 4. Last number in the last sentence
    if sentences:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", sentences[-1])
        if nums:
            v = parse_number(nums[-1])
            if v is not None:
                return v

    # 5. Last number anywhere
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        v = parse_number(nums[-1])
        if v is not None:
            return v

    return None

def extract_answer_any(text):
    """
    Best-effort extraction for the answer-distribution chips.

    Fallback chain (most reliable → least):
      1. \\boxed{...}
      2. explicit patterns ("Final Answer: X", "the answer is X", "has X left")
      3. math_verify — same extractor the judge uses
      4. last number in the last sentence
      5. last number anywhere
    """
    if not text:
        return None

    # 1. Boxed
    boxed = extract_boxed_last(text)
    if boxed is not None:
        v = parse_number(boxed)
        if v is not None:
            return v

    # 2. Explicit patterns
    for pat in ANSWER_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            v = parse_number(matches[-1])
            if v is not None:
                return v

    # 3. math_verify (same extraction the judge uses)
    try:
        pred = mv_parse(
            text,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
        )
        if pred:
            for elem in pred:
                if isinstance(elem, (int, float)):
                    return float(elem)
                v = parse_number(str(elem))
                if v is not None:
                    return v
    except Exception:
        pass

    # 4. Last number in the last sentence
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", sentences[-1])
        if nums:
            v = parse_number(nums[-1])
            if v is not None:
                return v

    # 5. Last number anywhere
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        v = parse_number(nums[-1])
        if v is not None:
            return v

    return None


def judge(completion, gold):
    if not completion:
        return False
    try:
        pred = mv_parse(completion,
                        extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()])
        gt_text = str(gold).strip()
        if "\\boxed" not in gt_text and "$" not in gt_text:
            gt_text = f"\\boxed{{{gt_text}}}"
        gt = mv_parse(gt_text,
                      extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        if pred and gt and verify(gt, pred):
            return True
# falls through to regex when verify says NO
    except Exception:
        pass
    p = extract_answer_regex(completion)
    g = parse_number(gold)
    if p is not None and g is not None:
        return abs(p - g) < 1e-4
    return False


# ============================================================
# UNBIASED PASS@K
# ============================================================

def unbiased_pass_at_k(n, c, k):
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_at_k_curve(n_correct, n_total, max_k=32):
    kmax = min(max_k, n_total)
    return [unbiased_pass_at_k(n_total, n_correct, k) for k in range(1, kmax + 1)]


# ============================================================
# FRESH STATS
# ============================================================

def extract_judge_view(text):
    """
    Same extractor order as judge(), plus a truncated-completion fallback.

    Fallback chain:
      1. math_verify  (same call the judge makes)
      2. regex patterns ("Final Answer: X", "the answer is X", "has X left", boxed)
      3. last number in the last sentence
      4. last number anywhere   ← new; catches truncated completions
    """
    if not text:
        return None

    # 1. math_verify
    try:
        pred = mv_parse(
            text,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
        )
        if pred:
            for elem in pred:
                if isinstance(elem, (int, float)):
                    return float(elem)
                v = parse_number(str(elem))
                if v is not None:
                    return v
    except Exception:
        pass

    # 2. Regex explicit patterns
    r = extract_answer_regex(text)
    if r is not None:
        return r

    # 3. Last number in the last sentence
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", sentences[-1])
        if nums:
            v = parse_number(nums[-1])
            if v is not None:
                return v

    # 4. Last number anywhere (catches truncated completions)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        v = parse_number(nums[-1])
        if v is not None:
            return v

    return None

def compute_fresh_stats(completions, gt):
    n = len(completions)
    trimmed = [trim_at_first_boxed(c) for c in completions]
    flags = [judge(t, gt) for t in trimmed]
    c = sum(flags)

    # Answer distribution — SAME extractor the judge uses
    answer_counts = {}
    unextracted = 0
    for t in trimmed:
        v = extract_judge_view(t)
        if v is None:
            unextracted += 1
        else:
            key = round(v, 4)
            answer_counts[key] = answer_counts.get(key, 0) + 1

    print(f"[stats] n={n} correct={c} extracted={len(answer_counts)} unextracted={unextracted}")

    pass_at_1 = (c / n) if n > 0 else 0.0
    p32 = unbiased_pass_at_k(n, c, 32)
    if p32 is None:
        p32 = unbiased_pass_at_k(n, c, n) or 0.0

    return {
        "n_correct": c,
        "n_total": n,
        "unique_answers": len(answer_counts),
        "answer_counts": answer_counts,
        "unextracted": unextracted,
        "pass_at_1": pass_at_1,
        "pass_at_32": p32,
    }


# ============================================================
# LOAD + BUILD
# ============================================================

@st.cache_data
def load_raw(path="demo_data_top32.json"):
    with open(path) as f:
        return json.load(f)


raw_data = load_raw()


@st.cache_data
def build_by_q(raw, _cache_version=8):
    out = {}
    for rec in raw:
        qidx = rec["question_index"]
        out.setdefault(qidx, {"question": rec["question"],
                              "gt": rec["gt"],
                              "models": {}})
        out[qidx]["models"][rec["model"]] = {
            "completions": rec["completions"],
            "stats": compute_fresh_stats(rec["completions"], rec["gt"]),
        }
    out = {
        qidx: q for qidx, q in out.items()
        if sum(m["stats"]["n_correct"] for m in q["models"].values()) > 0
    }
    return out


by_q = build_by_q(raw_data)


# ============================================================
# HELPERS
# ============================================================

def escape_html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def badge(ok, text=None):
    if text is None:
        text = "CORRECT" if ok else "WRONG"
    style = ("background:#dafbe1;color:#1a7f37;" if ok
             else "background:#ffebe9;color:#cf222e;")
    mark = "✓" if ok else "✗"
    return (
        f'<span style="display:inline-block;padding:2px 10px;'
        f'border-radius:10px;font-size:11px;font-weight:700;'
        f'letter-spacing:0.3px;{style}">{mark} {text}</span>'
    )


def fmt_number(v):
    try:
        f = float(v)
    except Exception:
        return str(v)
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:.3g}"


def answer_chips_html(answer_counts, gt=None, max_chips=6):
    if not answer_counts:
        return (
            '<div style="font-size:11px;color:#8c959f;'
            'font-style:italic;">no numeric answer extracted</div>'
        )

    gt_num = None
    if gt is not None:
        try:
            gt_num = round(float(str(gt).strip()), 4)
        except Exception:
            gt_num = None

    items = sorted(answer_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chips = []
    for val, cnt in items[:max_chips]:
        is_gt = (gt_num is not None and abs(val - gt_num) < 1e-4)
        if is_gt:
            bg, fg, border = "#dafbe1", "#1a7f37", "1px solid #2ea043"
        else:
            bg, fg, border = "#eaeef2", "#57606a", "1px solid transparent"
        chips.append(
            f'<span style="display:inline-block;padding:2px 7px;'
            f'margin:1px 3px 1px 0;background:{bg};border:{border};'
            f'border-radius:9px;font-size:11px;font-weight:600;'
            f'font-family:ui-monospace,monospace;color:{fg};">'
            f'{fmt_number(val)}'
            f'<span style="color:#8c959f;font-weight:500;margin-left:3px;">×{cnt}</span>'
            f'</span>'
        )
    if len(items) > max_chips:
        chips.append(
            f'<span style="font-size:11px;color:#8c959f;">'
            f'+{len(items) - max_chips} more</span>'
        )
    return "".join(chips)


def sparkline_svg(values, color="#0969da", height=48):
    if not values:
        return ""
    n = len(values)
    width = 240
    padding_l, padding_r = 4, 4
    padding_t, padding_b = 6, 12
    inner_w = width - padding_l - padding_r
    inner_h = height - padding_t - padding_b

    pts = []
    for i, v in enumerate(values):
        x = padding_l + (inner_w * i / (n - 1)) if n > 1 else padding_l + inner_w / 2
        y = padding_t + (inner_h * (1 - v))
        pts.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    baseline_y = padding_t + inner_h
    area_pts = f"{pts[0][0]:.1f},{baseline_y:.1f} " + polyline + \
               f" {pts[-1][0]:.1f},{baseline_y:.1f}"
    mid_y = padding_t + inner_h * 0.5

    svg = f'''
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}"
         style="display:block;margin-top:8px;" preserveAspectRatio="none">
      <line x1="{padding_l}" y1="{baseline_y:.1f}"
            x2="{width-padding_r}" y2="{baseline_y:.1f}"
            stroke="#e0e0e0" stroke-width="1"/>
      <line x1="{padding_l}" y1="{mid_y:.1f}"
            x2="{width-padding_r}" y2="{mid_y:.1f}"
            stroke="#f0f0f0" stroke-width="1" stroke-dasharray="2,3"/>
      <polygon points="{area_pts}" fill="{color}" opacity="0.10"/>
      <polyline points="{polyline}"
                fill="none" stroke="{color}" stroke-width="1.8"
                stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
    '''
    svg += (
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#8c959f;margin-top:-6px;padding:0 4px;">'
        f'<span>k=1</span><span>k={n}</span></div>'
    )
    return svg


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛 Controls")

qids = sorted(by_q.keys())
selected_q = st.sidebar.selectbox(
    "Question",
    qids,
    format_func=lambda q: f"Q{q} · GT={by_q[q]['gt']}",
)
n_samples = st.sidebar.slider("Samples per model", 1, 32, 8)

q = by_q[selected_q]


# ============================================================
# TABS
# ============================================================

tab_compare, tab_gate = st.tabs(["📊 Model Comparison", "🎯 Confidence Gate"])


# ============================================================
# TAB 1 — MODEL COMPARISON
# ============================================================

with tab_compare:
    st.markdown(
        f"""
        <div class="question-card">
          <div style="font-size: 22px; font-weight: 700;">
            Question {selected_q}
            <span class="gt-pill">GT = {escape_html(q['gt'])}</span>
          </div>
          <div style="margin-top: 10px; font-size: 15px; color: #1f2328;">
            {escape_html(q['question'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    model_order = ["base", "grpoV12_KL_step60", "Gated_V1_step60", "v12_step60"]
    model_labels = {
        "base": "Base (no training)",
        "grpoV12_KL_step60": "GRPO + KL",
        "Gated_V1_step60": "Gated DRL",
        "v12_step60": "DistilledRL V12",
    }
    model_colors = {
        "base": "#8c959f",
        "grpoV12_KL_step60": "#cf222e",
        "Gated_V1_step60": "#8250df",
        "v12_step60": "#0969da",
    }

    for col, model_name in zip(cols, model_order):
        rec = q["models"].get(model_name)
        if not rec:
            continue

        with col:
            stats = rec["stats"]
            p1 = stats["pass_at_1"]
            card_class = "correct" if p1 >= 0.5 else "wrong"

            st.markdown(
                f"""
                <div class="column-card {card_class}">
                  <div class="model-header">{model_labels[model_name]}</div>
                  <span class="stat-badge badge-grey">
                    {stats['n_correct']}/{stats['n_total']} correct
                  </span>
                  <span class="stat-badge badge-grey">
                    pass@32 = {int(round(stats['pass_at_32']*100))}%
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            curve = pass_at_k_curve(stats["n_correct"], stats["n_total"], max_k=32)
            st.markdown(
                f'<div class="section-label">pass@k curve</div>'
                f'{sparkline_svg(curve, color=model_colors[model_name])}',
                unsafe_allow_html=True,
            )

            label_suffix = f"{stats['unique_answers']} unique"
            if stats.get("unextracted", 0) > 0:
                label_suffix += f" · {stats['unextracted']} unextracted"
            
            chips_html = answer_chips_html(stats["answer_counts"], gt=q["gt"], max_chips=6)
            st.markdown(
                f'<div class="section-label">answers seen ({label_suffix})</div>'
                f'<div style="margin-bottom:8px;">{chips_html}</div>',
                unsafe_allow_html=True,
            )

            for i, comp in enumerate(rec["completions"][:n_samples]):
                trimmed = trim_at_first_boxed(comp)
                ok = judge(trimmed, q["gt"])
                extracted = extract_boxed_last(trimmed)

                icon = "🟢" if ok else "🔴"
                with st.expander(f"{icon} Sample {i+1}", expanded=(i < 2)):
                    st.markdown(badge(ok), unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="sample-wrap {"correct" if ok else "wrong"}">'
                        f'<div class="reasoning">{escape_html(trimmed)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if extracted:
                        st.caption(f"Extracted: `{extracted}`")


# ============================================================
# TAB 2 — CONFIDENCE GATE
# ============================================================

with tab_gate:
    @st.cache_data
    def load_gate_heatmap(path="gate_heatmap_data.json"):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    gate_data = load_gate_heatmap()

    if gate_data is None:
        st.info(
            "Gate heatmap data not found. Run `capture_gate_heatmap.py` and place "
            "`gate_heatmap_data.json` in the same directory as this app."
        )
    else:
        st.subheader("Where the teacher was trusted, token by token")
    

        # Show the heatmap's OWN question + GT (independent of the dropdown)
        st.markdown(
            f"""
            <div class="heatmap-question">
              <b>Captured example</b> — this tab shows a fixed, pre-recorded
              completion. It is independent of the dropdown above.
              <br/><br/>
              <b>Q{gate_data['question_index']}</b>
              &nbsp;·&nbsp; <b>GT = <code>{escape_html(str(gate_data['gt']))}</code></b>
              <br/>
              <span style="color:#6e5600;">
                {escape_html(gate_data['question'])}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="gate-explanation">
              Each token is colored by its gate value
              <code>c = σ((H* − H) / τ)</code>.
              <b>Bright blue</b> = teacher fully trusted (low entropy, teacher is confident).
              <b>Faint blue</b> = teacher silenced (high entropy, teacher is unsure).
              <br/>
              Calibration: <code>H* = {gate_data['entropy_threshold']}</code>,
              <code>τ = {gate_data['gate_temp']}</code> (from T=1.2 rollouts).
              Hover any token to see its entropy and gate value.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="heatmap-legend">
              <span>c ≈ 0<br/><span style="font-size:11px;">(teacher ignored)</span></span>
              <span class="heatmap-swatch"></span>
              <span>c ≈ 1<br/><span style="font-size:11px;">(full teacher)</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Render tokens: split by line so \n shows as a real line break ---
        def render_token(t):
            g = float(t["gate"])
            bg_alpha = 0.05 + 0.90 * g
            bg = f"rgba(88,166,255,{bg_alpha:.3f})"
            fg = "#1f2328" if g < 0.75 else "#ffffff"
            text = t["text"]
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Replace leading/trailing spaces with nbsp so they show
            text = text.replace(" ", "&nbsp;")
            text = text.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
            if not text:
                text = "&nbsp;"
            tooltip = f"H={t['entropy']:.2f}, c={g:.2f}"
            return (
                f'<span class="tok" style="background:{bg};color:{fg};" '
                f'title="{tooltip}">{text}</span>'
            )

        # Group tokens into lines using \n in their text
        lines_html = []
        current_line = []
        for t in gate_data["tokens"]:
            raw = t["text"]
            if "\n" in raw:
                # split on \n; emit current_line, then start new line
                parts = raw.split("\n")
                for k, part in enumerate(parts):
                    if part:
                        # keep original gate/entropy on the fragment
                        frag = dict(t)
                        frag["text"] = part
                        current_line.append(render_token(frag))
                    if k < len(parts) - 1:
                        # flush current line
                        lines_html.append(
                            '<span class="heatmap-line">' +
                            "".join(current_line) +
                            '</span>'
                        )
                        current_line = []
            else:
                current_line.append(render_token(t))
        if current_line:
            lines_html.append(
                '<span class="heatmap-line">' +
                "".join(current_line) +
                '</span>'
            )

        html = "".join(lines_html) if lines_html else "&nbsp;"
        st.markdown(
            f'<div class="heatmap-container">{html}</div>',
            unsafe_allow_html=True,
        )

        gates = [float(t["gate"]) for t in gate_data["tokens"]]
        n = len(gates)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tokens", f"{n}")
        c2.metric("Mean gate", f"{sum(gates)/n:.3f}")
        c3.metric("Fraction c > 0.9", f"{sum(1 for g in gates if g > 0.9)/n*100:.1f}%")
        c4.metric("Fraction c < 0.2", f"{sum(1 for g in gates if g < 0.2)/n*100:.1f}%")

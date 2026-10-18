r"""
copilot_core_v2.py
Cockpit VQA core for model v2 (CockpitNetV2, dataset_v2). Same public interface
as cockpit/gui/copilot_core.py's CopilotSystem -- answer(image, question) returns
{params, phase, intent, confidence, answer, timings} -- so integrated.py can use
either. The v1 file is left untouched.

What changes vs v1 (copilot_core.py):
  * model + preprocessing come from cockpit_model_v2.py (load_checkpoint, decode,
    eval_transform) -- the SAME code used in training, so they cannot drift apart
  * auto-crop: a full FlightGear screenshot (16:10, the capture setup) is cropped
    to the instrument panel with the capture script's exact screen fractions;
    an already-cropped panel image (~936x729 shape) is used as is
  * panel check: the crop is compared with dataset_v2/reference_panel.png (the
    same NCC check the capture script used); a low score is reported as a
    warning instead of silently answering about an image the model never saw
  * continuous readings (no bins); phase comes from the LEARNED head, with its
    probability, instead of the hand-written rule
  * two new intents: vertical speed and engine RPM
The intent router (Qwen2.5-1.5B gate -> TF-IDF -> substring) is the v1 design,
copied here so cockpit2 stays self-contained.
"""

import math
import os
import re
import time

import numpy as np
import torch
from PIL import Image

from cockpit_model_v2 import decode, eval_transform, load_checkpoint

# Capture-time crop, as screen fractions (capture_synthetic_v2.py --crop default).
PANEL_CROP = (0.365, 0.465, 0.690, 0.870)
PANEL_ASPECT = 936 / 729          # shape of the training crops (~1.284)
SCREEN_ASPECT = 2880 / 1800       # capture screen (16:10)
PANEL_SIM_WARN = 0.55             # capture script rejected frames below this


# =========================================================================
# Image handling (no torch needed -> testable on its own)
# =========================================================================
def to_panel(img, crop=PANEL_CROP, tol=0.06):
    """Returns (panel_image, how) where how is 'already_panel', 'cropped_screenshot'
    or 'unknown_shape' (then the whole image is used and a warning is shown)."""
    w, h = img.size
    ar = w / h
    if abs(ar - PANEL_ASPECT) / PANEL_ASPECT < tol:
        return img, "already_panel"
    if abs(ar - SCREEN_ASPECT) / SCREEN_ASPECT < tol:
        box = (round(crop[0] * w), round(crop[1] * h), round(crop[2] * w), round(crop[3] * h))
        return img.crop(box), "cropped_screenshot"
    return img, "unknown_shape"


def _small_gray(img):
    return np.asarray(img.convert("L").resize((96, 75)), dtype=np.float32)


def panel_similarity(img, ref_gray):
    """Normalised cross-correlation with the reference panel (same as capture)."""
    a = _small_gray(img)
    a = a - a.mean()
    b = ref_gray
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


# =========================================================================
# Answers (no torch needed -> testable on its own)
# =========================================================================
PHASE_TEXT = {
    "engine_off": "parked with the engine off",
    "taxi": "taxiing on the ground",
    "takeoff_roll": "on the takeoff roll, accelerating down the runway",
    "climb": "climbing",
    "level_flight": "in level flight",
}
NOT_APPLICABLE = (
    "Not applicable — I couldn't confidently match that question to something I can answer. "
    "I can answer questions about altitude, airspeed, heading, bank/roll, pitch/climbing, "
    "vertical speed, engine RPM and flight phase.")


def make_answer(intent, p, phase, phase_conf, score, threshold=0.35):
    if intent is None or score < threshold:
        return NOT_APPLICABLE
    if intent == "get_altitude":
        return f"The altimeter reads about {p['altitude']:,.0f} ft."
    if intent == "get_airspeed":
        if p["airspeed"] < 40:
            return (f"The airspeed indicator reads about {max(p['airspeed'], 0):.0f} kt. Note: the "
                    f"C172 airspeed scale has no markings below ~40 kt, so low speeds are less precise.")
        return f"The airspeed is about {p['airspeed']:.0f} kt."
    if intent == "get_heading":
        return f"The heading indicator shows about {p['heading']:03.0f}°."
    if intent == "is_banking":
        roll = p["roll"]
        if abs(roll) < 2:
            return f"The wings are approximately level (bank {roll:+.1f}°)."
        side = "left" if roll < 0 else "right"
        return f"Yes — the aircraft is banked about {abs(roll):.0f}° to the {side}."
    if intent == "is_climbing":
        vs, pitch = p["vs"], p["pitch"]
        if phase in ("engine_off", "taxi"):
            return "No — the aircraft is on the ground."
        if phase == "takeoff_roll":
            return "Not yet — the aircraft is on the takeoff roll and hasn't started climbing."
        if vs > 200:
            return f"Yes — climbing at about {vs:.0f} ft/min with the nose about {pitch:.0f}° up."
        if vs < -200:
            return f"No — the aircraft is descending at about {abs(vs):.0f} ft/min."
        return f"No — vertical speed is about {vs:+.0f} ft/min, so the aircraft is roughly level."
    if intent == "get_vertical_speed":
        return f"The vertical speed indicator reads about {p['vs']:+.0f} ft/min."
    if intent == "get_rpm":
        if p["rpm"] < 200:
            return "The tachometer shows the engine is not running (about 0 RPM)."
        return f"The engine is turning about {p['rpm']:,.0f} RPM."
    if intent == "flight_phase":
        return (f"The aircraft appears to be {PHASE_TEXT.get(phase, 'in an unclear phase')} "
                f"(model confidence {phase_conf * 100:.0f}%).")
    return "I recognised the topic but don't have an answer template for it yet."


# =========================================================================
# System
# =========================================================================
class CopilotSystemV2:
    INTENT_DESCRIPTIONS = {
        "get_altitude": "the question asks about altitude, height, or elevation",
        "get_airspeed": "the question asks about airspeed, speed, or velocity",
        "get_heading": "the question asks about heading, compass direction, or bearing",
        "is_banking": "the question asks about banking, rolling, turning, or wing tilt",
        "is_climbing": "the question asks whether the aircraft is climbing, descending, "
                       "ascending, or about its pitch / nose attitude",
        "get_vertical_speed": "the question asks about vertical speed or rate of climb or descent "
                              "(feet per minute)",
        "get_rpm": "the question asks about engine RPM, engine speed, the tachometer, "
                   "or whether the engine is running",
        "flight_phase": "the question asks about the overall flight phase, status, "
                        "or what the aircraft is currently doing",
    }
    INTENT_EXAMPLES = {   # only for the TF-IDF / substring fallback tiers
        "get_altitude": ["what is the altitude", "how high are we", "altimeter reading",
                         "what altitude are we at", "how many feet are we at"],
        "get_airspeed": ["what is the airspeed", "how fast are we going", "current speed",
                         "how many knots are we flying at"],
        "get_heading": ["what heading are we on", "which direction are we flying",
                        "current compass heading", "what's our heading"],
        "is_banking": ["are we banking", "is the plane turning", "are the wings level",
                       "what's our current bank angle", "are we rolling left or right"],
        "is_climbing": ["are we climbing", "are we gaining altitude", "is the nose up",
                        "are we descending", "what is the pitch"],
        "get_vertical_speed": ["what is the vertical speed", "rate of climb",
                               "how many feet per minute", "what does the vsi show"],
        "get_rpm": ["what is the engine rpm", "engine speed", "tachometer reading",
                    "is the engine running"],
        "flight_phase": ["what phase of flight are we in", "what's our current status",
                         "are we on the ground or in the air", "is this takeoff, climb, or cruise"],
    }

    def __init__(self, model_dir, reference_panel=None, confidence_threshold=0.35,
                 llm_on_cpu=False, use_llm=True):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.path.join(model_dir, "best_model.pt")
        self.model, ckpt = load_checkpoint(ckpt_path, self.device)
        self.norm = ckpt["norm"]
        self.checkpoint_epoch = ckpt.get("epoch")
        self.model_info = (f"CockpitNetV2 (ResNet18), epoch {ckpt.get('epoch')}, "
                           f"alt_fine={ckpt.get('alt_fine', False)}, alt_short={ckpt.get('alt_short', False)}")
        self.transform = eval_transform()
        self.ref_gray = None
        if reference_panel and os.path.exists(reference_panel):
            g = _small_gray(Image.open(reference_panel))
            self.ref_gray = g - g.mean()
        else:
            print(f"WARNING: reference panel not found ({reference_panel}) -- panel check disabled.")
        self.llm_device = torch.device("cpu") if llm_on_cpu else self.device
        self._init_router(use_llm)

    # ---------------------------------------------------------------- router (v1 design)
    def _init_router(self, use_llm):
        self.have_llm_gate = self.have_tfidf = False
        if use_llm:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                name = "Qwen/Qwen2.5-1.5B-Instruct"
                self._tok = AutoTokenizer.from_pretrained(name)
                dtype = torch.float16 if self.llm_device.type == "cuda" else torch.float32
                self._llm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
                self._llm.to(self.llm_device).eval()
                lines = ["You are a strict classifier for a cockpit flight-instrument question-answering "
                         "assistant.", "Given a user's question, decide which ONE of the following "
                         "categories it belongs to, and respond with EXACTLY ONE WORD: the category name, "
                         "and nothing else.", ""]
                lines += [f"{k}: {v}" for k, v in self.INTENT_DESCRIPTIONS.items()]
                lines += ["IRRELEVANT: the question is not about any of the above (weather, small talk, "
                          "general knowledge, anything unrelated to this aircraft's instruments)", "",
                          "Respond with only the single category word."]
                self._sys_prompt = "\n".join(lines)
                self.have_llm_gate = True
            except Exception as e:   # ImportError, no network on first run, out of GPU memory...
                print(f"WARNING: LLM intent gate unavailable ({type(e).__name__}: {e}) -- using TF-IDF.")
        if not self.have_llm_gate:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                texts, labels = [], []
                for k, exs in self.INTENT_EXAMPLES.items():
                    texts += exs
                    labels += [k] * len(exs)
                self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                self._tfidf_m = self._tfidf.fit_transform(texts)
                self._tfidf_labels = labels
                self.have_tfidf = True
            except ImportError:
                print("WARNING: scikit-learn missing -- falling back to substring matching (poor quality).")

    def _llm_classify(self, question):
        msgs = [{"role": "system", "content": self._sys_prompt}, {"role": "user", "content": question}]
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self.llm_device)
        with torch.no_grad():
            out = self._llm.generate(**inputs, max_new_tokens=8, do_sample=False,
                                     pad_token_id=self._tok.eos_token_id)
        text = self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
        # longest names first so e.g. "get_vertical_speed" is not matched as something shorter
        for label in sorted(list(self.INTENT_DESCRIPTIONS) + ["irrelevant"], key=len, reverse=True):
            if label in text:
                return None if label == "irrelevant" else label
        return "PARSE_FAILED"

    @staticmethod
    def keyword_fallback(question):
        q = question.lower()
        checks = [("get_vertical_speed", ["vertical speed", "feet per minute", "fpm", "rate of climb", "vsi"]),
                  ("get_rpm", ["rpm", "engine", "tachometer", "rev"]),
                  ("get_altitude", ["altitude", "height", "elevation", "altimeter", "how high"]),
                  ("get_airspeed", ["airspeed", "speed", "knots", "fast"]),
                  ("get_heading", ["heading", "direction", "compass", "bearing"]),
                  ("is_banking", ["bank", "roll", "tilt", "turn", "wings"]),
                  ("is_climbing", ["climb", "ascend", "descend", "pitch", "nose"]),
                  ("flight_phase", ["phase", "status", "takeoff", "landing", "cruise", "ground"])]
        for intent, words in checks:
            if any(w in q for w in words):
                return intent
        return None

    def route_question(self, question):
        t0 = time.time()
        if self.have_llm_gate:
            label = self._llm_classify(question)
            if label == "PARSE_FAILED":
                label = self.keyword_fallback(question)
                return label, (0.5 if label else 0.0), (time.time() - t0) * 1000
            return label, (1.0 if label else 0.0), (time.time() - t0) * 1000
        if self.have_tfidf:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(self._tfidf.transform([question]), self._tfidf_m)[0]
            i = int(sims.argmax())
            score = float(sims[i])
            return (self._tfidf_labels[i] if score > 0 else None), score, (time.time() - t0) * 1000
        label = self.keyword_fallback(question)
        return label, (0.5 if label else 0.0), (time.time() - t0) * 1000

    # ---------------------------------------------------------------- perception
    def run_perception(self, image_path_or_pil):
        t0 = time.time()
        img = (Image.open(image_path_or_pil) if isinstance(image_path_or_pil, str)
               else image_path_or_pil).convert("RGB")
        panel, how = to_panel(img)
        sim = panel_similarity(panel, self.ref_gray) if self.ref_gray is not None else None
        x = self.transform(panel).unsqueeze(0).to(self.device)
        pre_ms = (time.time() - t0) * 1000
        t0 = time.time()
        with torch.no_grad():
            row = decode(self.model(x), self.norm)[0]
        inf_ms = (time.time() - t0) * 1000
        warnings = []
        if how == "unknown_shape":
            warnings.append("Image shape matches neither a full FlightGear screenshot (16:10) nor the "
                            "instrument-panel crop, so it was used as is. Readings may be wrong.")
        if sim is not None and sim < PANEL_SIM_WARN:
            warnings.append(f"The panel doesn't look like the training view (similarity {sim:.2f} < "
                            f"{PANEL_SIM_WARN}). The model was trained only on the C172 panel in "
                            f"FlightGear's default cockpit view, so readings may be unreliable.")
        return row, {"panel": panel, "crop_mode": how, "panel_similarity": sim, "warnings": warnings,
                     "preprocess_ms": pre_ms, "inference_ms": inf_ms}

    def answer(self, image_path_or_pil, question):
        p, info = self.run_perception(image_path_or_pil)
        intent, score, routing_ms = self.route_question(question)
        text = make_answer(intent, p, p["phase"], p["phase_confidence"], score, self.confidence_threshold)
        if info["warnings"]:
            text += "\n\n⚠️ " + "\n\n⚠️ ".join(info["warnings"])
        return {
            "params": p, "phase": p["phase"], "phase_confidence": p["phase_confidence"],
            "intent": intent, "confidence": score, "answer": text,
            "panel": info["panel"], "crop_mode": info["crop_mode"],
            "panel_similarity": info["panel_similarity"], "warnings": info["warnings"],
            "timings": {"preprocess_ms": info["preprocess_ms"], "inference_ms": info["inference_ms"],
                        "routing_ms": routing_ms,
                        "total_ms": info["preprocess_ms"] + info["inference_ms"] + routing_ms},
        }

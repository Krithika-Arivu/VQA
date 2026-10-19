r"""
integrated.py
Single Gradio app for the Visual Co-Pilot project. Replaces the earlier
Tkinter-launcher-that-spawns-a-subprocess design (fragile: guessed a fixed
wait time before opening the browser, and could hit a connection-refused
page if the model hadn't finished loading yet).

One process, one URL, one thing to wait on (Gradio's own "Running on
local URL" message) -- same as running cockpit/gui/app.py or radar/app.py
directly today.

FLOW:
  1. You run this once, with startup args for whichever module(s) you plan
     to use (see --help). Nothing is loaded yet at this point.
  2. One browser tab opens to a landing view: two cards, Cockpit and Radar.
  3. Clicking a card LAZY-LOADS that module's system (CopilotSystem or
     RadarSystem) the first time only, then switches to that module's real
     working panel -- drag-and-drop image box + question box + answer box,
     same UI cockpit/gui/app.py already has. No folder-picker dialogs.
  4. A "<- Back" button on each panel returns to the landing view. Systems
     stay loaded in memory once loaded (no reload switching back and forth).

Setup:
  pip install gradio torch torchvision pandas pillow sentence-transformers --break-system-packages
  (plus radar/requirements-radar.txt's deps, e.g. YOLOv5's requirements --
   both modules' dependencies now need to coexist in ONE Python environment,
   since this is one process, not two separate subprocesses like before)

Run FROM THE REPO ROOT (needs cockpit2/ (and/or cockpit/) and radar/ next to it):
  python integrated.py --radar_base_dir "C:\...\radar_base"
      -> Cockpit uses model v2 by default (cockpit2/dataset_v2/model_v2/best_model.pt)
  python integrated.py --cockpit_version v1 --cockpit_data_dir "C:\...\consolidated" ...
      -> old v1 model (kept for comparison only)

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

If you only plan to use one module this run, you can omit the other
module's required arg -- that card will just show a clear error (not a
crash) if you click it without its data configured.
"""

import os
import sys
import argparse

import gradio as gr
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
COCKPIT_DIR = os.path.join(REPO_ROOT, "cockpit", "gui")
COCKPIT2_DIR = os.path.join(REPO_ROOT, "cockpit2", "gui")
RADAR_DIR = os.path.join(REPO_ROOT, "radar")

# Both modules do local, same-folder imports (`from copilot_core import
# CopilotSystem`, `from radar_system import RadarSystem`). Since everything
# now runs in ONE process (unlike the old subprocess-per-module launcher),
# both folders need to be on sys.path so those local imports resolve no
# matter which one loads first.
for _p in (COCKPIT_DIR, COCKPIT2_DIR, RADAR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser()
# Cockpit model choice. v2 = CockpitNetV2 on dataset_v2 (default); v1 = old CockpitNet.
parser.add_argument("--cockpit_version", choices=["v1", "v2"], default="v2")
parser.add_argument("--cockpit2_model_dir",
                    default=os.path.join(REPO_ROOT, "cockpit2", "dataset_v2", "model_v2"),
                    help="v2: folder containing best_model.pt")
parser.add_argument("--cockpit2_reference",
                    default=os.path.join(REPO_ROOT, "cockpit2", "dataset_v2", "reference_panel.png"),
                    help="v2: reference panel image for the 'is this the training view?' check")
parser.add_argument("--cockpit_llm_on_cpu", action="store_true",
                    help="v2: run the Qwen intent router on CPU (use if the GPU runs out of memory)")
parser.add_argument("--cockpit_no_llm", action="store_true",
                    help="v2: skip the Qwen router, use the TF-IDF fallback (fast start, no download)")
# v1 cockpit args -- same names/defaults as cockpit/gui/app.py's own parser.
parser.add_argument("--cockpit_data_dir", default=None,
                     help="v1 only: your local consolidated/ folder.")
parser.add_argument("--cockpit_img_width", type=int, default=224)
parser.add_argument("--cockpit_img_height", type=int, default=224)
parser.add_argument("--cockpit_use_letterbox", action="store_true")
parser.add_argument("--cockpit_confidence_threshold", type=float, default=0.35)
# Radar args -- same names/defaults as radar/app.py's own parser.
parser.add_argument("--radar_base_dir", default=None,
                     help="Your local radar_base/ folder (models/, data/, logs/). "
                          "Required to use the Radar card.")
parser.add_argument("--radar_repo_root", default=None)
parser.add_argument("--radar_yolo_root", default=None)
parser.add_argument("--radar_yolo_conf", type=float, default=0.25)
parser.add_argument("--radar_cnn_checkpoint", default="cnn_best_model.pth")
parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
args = parser.parse_args()

# =========================================================================
# LAZY LOADING -- nothing below actually loads a model until the matching
# "Launch" button is clicked for the first time.
# =========================================================================
_cockpit_system = None
_radar_system = None


def get_cockpit_system():
    global _cockpit_system
    if _cockpit_system is not None:
        return _cockpit_system, None
    if args.cockpit_version == "v2":
        ckpt = os.path.join(args.cockpit2_model_dir, "best_model.pt")
        if not os.path.exists(ckpt):
            return None, (f"Cockpit v2 model not found at {ckpt} -- train it (cockpit2/process/"
                           f"train_v2.py) or pass --cockpit2_model_dir.")
        from copilot_core_v2 import CopilotSystemV2
        print(f"Loading Cockpit v2 from {ckpt} ...")
        _cockpit_system = CopilotSystemV2(
            model_dir=args.cockpit2_model_dir,
            reference_panel=args.cockpit2_reference,
            confidence_threshold=args.cockpit_confidence_threshold,
            llm_on_cpu=args.cockpit_llm_on_cpu,
            use_llm=not args.cockpit_no_llm,
        )
        print(f"Cockpit v2 loaded: {_cockpit_system.model_info}")
        return _cockpit_system, None
    if not args.cockpit_data_dir:
        return None, ("Cockpit isn't configured for this run -- restart with "
                       "--cockpit_data_dir pointing at your consolidated/ folder.")
    from copilot_core import CopilotSystem
    print("Loading Cockpit (CopilotSystem)...")
    _cockpit_system = CopilotSystem(
        data_dir=args.cockpit_data_dir,
        img_width=args.cockpit_img_width,
        img_height=args.cockpit_img_height,
        use_letterbox=args.cockpit_use_letterbox,
        confidence_threshold=args.cockpit_confidence_threshold,
    )
    print(f"Cockpit loaded (checkpoint epoch {_cockpit_system.checkpoint_epoch}).")
    return _cockpit_system, None


def get_radar_system():
    global _radar_system
    if _radar_system is not None:
        return _radar_system, None
    if not args.radar_base_dir:
        return None, ("Radar isn't configured for this run -- restart with "
                       "--radar_base_dir pointing at your radar_base/ folder.")
    from radar_system import RadarSystem
    print(f"Loading Radar (checkpoint: {args.radar_cnn_checkpoint})...")
    try:
        _radar_system = RadarSystem(
            base_dir=args.radar_base_dir,
            repo_root=args.radar_repo_root,
            yolo_root=args.radar_yolo_root,
            yolo_conf=args.radar_yolo_conf,
            cnn_checkpoint=args.radar_cnn_checkpoint,
        )
    except FileNotFoundError as exc:
        return None, f"Radar model files missing: {exc}"
    print("Radar loaded.")
    return _radar_system, None


# =========================================================================
# COCKPIT PANEL CALLBACKS (mirrors cockpit/gui/app.py's run_analysis)
# =========================================================================
COCKPIT_PRESET_QUESTIONS = [
    "What is the current altitude?",
    "How fast are we going?",
    "What heading are we on?",
    "Are we banking left or right?",
    "Are we climbing?",
    "What phase of flight are we in?",
]
if args.cockpit_version == "v2":
    COCKPIT_PRESET_QUESTIONS += ["What is the vertical speed?", "What is the engine RPM?"]


def cockpit_redo_voice():
    """Clear the question box so a new question can be spoken."""
    return ""


COCKPIT_OUT_OF_SCOPE = (
    "**Not in scope.** I can only answer questions about this cockpit's instruments: altitude, "
    "airspeed, heading, bank/roll, pitch/climbing, vertical speed, engine RPM and flight phase.")


def cockpit_run_analysis(image, question):
    system, err = get_cockpit_system()
    if err:
        return err, "", "", 0.0, None
    if image is None:
        return "Please upload a cockpit panel image first.", "", "", 0.0, None
    if not question or not question.strip():
        return "Please enter or select a question.", "", "", 0.0, None

    result = system.answer(image, question)
    # Out-of-scope guard: the router found no cockpit intent (or too weak a match). Don't show
    # instrument readings that nobody asked for -- say plainly that the question is out of scope.
    if result["intent"] is None or result["confidence"] < args.cockpit_confidence_threshold:
        footer_md = (f"**Latency:** {result['timings']['total_ms']:.0f}ms  •  "
                     f"**Matched intent:** none (router confidence {result['confidence'] * 100:.0f}%)")
        return COCKPIT_OUT_OF_SCOPE, "", footer_md, result["confidence"] * 100, None
    p = result["params"]
    if args.cockpit_version == "v2":
        sim = result["panel_similarity"]
        key_detections_md = (
            f"- **Altitude:** {p['altitude']:,.0f} ft\n"
            f"- **Airspeed:** {max(p['airspeed'], 0):.0f} kt\n"
            f"- **Heading:** {p['heading']:03.0f}°\n"
            f"- **Pitch:** {p['pitch']:+.1f}°  •  **Roll:** {p['roll']:+.1f}°\n"
            f"- **Vertical speed:** {p['vs']:+.0f} ft/min  •  **Turn rate:** {p['turn']:+.2f} std-rate\n"
            f"- **Engine:** {max(p['rpm'], 0):,.0f} RPM\n"
            f"- **Flight phase (learned):** {result['phase']} ({result['phase_confidence'] * 100:.0f}%)\n"
            f"- **Input:** {result['crop_mode'].replace('_', ' ')}"
            + (f", panel match {sim:.2f}" if sim is not None else "")
        )
        footer_md = (
            f"**Model:** {system.model_info}  •  "
            f"**Latency:** {result['timings']['total_ms']:.0f}ms  •  "
            f"**Matched intent:** {result['intent']}"
        )
        return (result["answer"], key_detections_md, footer_md, result["confidence"] * 100,
                result["panel"])
    key_detections_md = (
        f"- **Altitude:** {p['altitude']:.0f} ft\n"
        f"- **Airspeed:** {p['airspeed']:.0f} kt\n"
        f"- **Heading:** {p['heading']:.0f}°\n"
        f"- **Pitch:** {p['pitch_bin']}° bin (~{p['pitch_value']:.0f}°)\n"
        f"- **Roll:** {p['roll_bin']}° bin (~{p['roll_value']:.0f}°)\n"
        f"- **Inferred phase:** {result['phase']} _(heuristic, not a certified classifier)_"
    )
    footer_md = (
        f"**Model:** CockpitNet (ResNet18)  •  "
        f"**Latency:** {result['timings']['total_ms']:.0f}ms  •  "
        f"**Matched intent:** {result['intent']}"
    )
    return result["answer"], key_detections_md, footer_md, result["confidence"] * 100, None


# =========================================================================
# RADAR PANEL CALLBACKS (mirrors radar/app.py's answer_question)
# =========================================================================
RADAR_PRESET_QUESTIONS = [
    "How many aircraft are visible on the radar?",
    "Is there any aircraft on the left side of the radar?",
    "Is there any aircraft on the right side of the radar?",
    "What is the callsign of the aircraft with the highest altitude?",
    "Which quadrant of the radar has the most aircraft?",
]


def radar_run_analysis(image_path, question):
    system, err = get_radar_system()
    if err:
        return err, None
    if not image_path:
        return "Please upload a radar image first.", None
    if not question or not question.strip():
        return "Please enter or select a question.", None

    result = system.handle_question(image_path, question)
    return result["answer"], result.get("image")


# =========================================================================
# GUI -- one landing view + two panels, toggled by visibility (no
# subprocess, no new server, no port/timing guesswork).
# =========================================================================
CUSTOM_CSS = """
.gradio-container {background-color: #0b0f19 !important;}
#title {color: #7c9cff; font-size: 1.6em; font-weight: 700; margin-bottom: 0.2em;}
#answer_box {background-color: #131824; border: 1px solid #232b3d; border-radius: 10px; padding: 16px;}
#detections_box {background-color: #131824; border: 1px solid #232b3d; border-radius: 10px; padding: 16px;}
#footer_box {color: #8891a5; font-size: 0.85em; margin-top: 8px;}
.launch-card {text-align: center;}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="blue")) as demo:
    gr.Markdown("### ✈️ Visual Co-Pilot", elem_id="title")

    # ---------------- Landing view ----------------
    with gr.Column(visible=True) as landing_view:
        gr.Markdown("Choose a module to launch.")
        with gr.Row():
            with gr.Column(elem_classes="launch-card"):
                gr.Markdown("**Cockpit**\n\nAltitude, airspeed, heading, bank, pitch, vertical speed, "
                            "engine RPM, flight phase." if args.cockpit_version == "v2" else
                            "**Cockpit**\n\nAltitude, airspeed, heading, banking, climbing, flight phase.")
                cockpit_card_btn = gr.Button("Launch Cockpit", variant="primary")
            with gr.Column(elem_classes="launch-card"):
                gr.Markdown("**Radar**\n\nAircraft count, position, altitude, callsign.")
                radar_card_btn = gr.Button("Launch Radar", variant="primary")

    # ---------------- Cockpit panel ----------------
    with gr.Column(visible=False) as cockpit_view:
        cockpit_back_btn = gr.Button("← Back")
        with gr.Row():
            with gr.Column(scale=1):
                cockpit_image_input = gr.Image(type="pil", label="Upload cockpit image (full FlightGear "
                                                "screenshot or instrument-panel crop)",
                                                sources=["upload", "clipboard"])
                cockpit_preset_dropdown = gr.Dropdown(
                    choices=COCKPIT_PRESET_QUESTIONS, label="Sample questions", value=None)
                cockpit_question_box = gr.Textbox(
                    label="Question", placeholder="Ask a question about the cockpit instruments...",
                    lines=2)
                cockpit_speak_btn = gr.Button("🎤 Speak your question", elem_id="cockpit_speak_btn")
                cockpit_redo_btn = gr.Button("🔄 Redo recording")
                cockpit_run_btn = gr.Button("Run Analysis", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("**Analysis Result**")
                cockpit_answer_output = gr.Markdown(elem_id="answer_box")
                with gr.Row():
                    cockpit_tts_checkbox = gr.Checkbox(label="🔊 Read answer aloud", value=True)
                    cockpit_stop_speech_btn = gr.Button("⏹ Stop", size="sm", scale=0)
                cockpit_confidence_output = gr.Slider(
                    label="Routing confidence", minimum=0, maximum=100, value=0, interactive=False)
                gr.Markdown("**Key Detections**")
                cockpit_detections_output = gr.Markdown(elem_id="detections_box")
                cockpit_footer_output = gr.Markdown(elem_id="footer_box")
                cockpit_panel_output = gr.Image(label="What the model read (instrument panel)",
                                                interactive=False)

        cockpit_preset_dropdown.change(fn=lambda q: q, inputs=cockpit_preset_dropdown,
                                        outputs=cockpit_question_box)
        cockpit_run_btn.click(
            fn=cockpit_run_analysis,
            inputs=[cockpit_image_input, cockpit_question_box],
            outputs=[cockpit_answer_output, cockpit_detections_output,
                     cockpit_footer_output, cockpit_confidence_output, cockpit_panel_output],
        )

        # Voice input: the browser's built-in speech recognition (Web Speech API, Chrome/Edge)
        # writes the spoken question into the question box; nothing is answered until the user
        # clicks Run Analysis. Redo clears the question box so a new one can be spoken.
        cockpit_speak_btn.click(
            fn=None, inputs=[cockpit_question_box], outputs=[cockpit_question_box],
            js=r"""
            (current) => new Promise((resolve) => {
                const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (!SR) {
                    alert('Voice input needs Chrome or Edge (speech recognition is not supported in this browser).');
                    resolve(current); return;
                }
                const btn = document.querySelector('#cockpit_speak_btn');
                const label = btn ? btn.textContent : '';
                const done = (value) => { if (btn) { btn.textContent = label; } resolve(value); };
                const rec = new SR();
                rec.lang = 'en-US';
                rec.interimResults = false;
                rec.continuous = false;
                rec.maxAlternatives = 1;
                let heard = null;
                rec.onresult = (e) => { heard = e.results[0][0].transcript.trim(); };
                rec.onerror = (e) => {
                    const msg = {
                        'not-allowed': 'Microphone access was blocked. Allow the microphone for this page and try again.',
                        'service-not-allowed': 'Microphone access was blocked. Allow the microphone for this page and try again.',
                        'no-speech': 'I could not hear a question. Click Speak and try again.',
                        'audio-capture': 'No microphone was found.',
                        'network': 'Speech recognition needs an internet connection.'
                    }[e.error] || ('Voice input error: ' + e.error);
                    alert(msg);
                };
                rec.onend = () => { done(heard ? heard : current); };
                if (btn) { btn.textContent = '🎙 Listening... speak now'; }
                try { rec.start(); } catch (err) { alert('Voice input error: ' + err); done(current); }
            })
            """,
        )
        cockpit_redo_btn.click(fn=cockpit_redo_voice, outputs=[cockpit_question_box])

        # Voice output: browser SpeechSynthesis (no cost, no extra Python dependency), fired
        # whenever the answer text changes - typed, preset or spoken question alike.
        cockpit_answer_output.change(
            fn=None,
            inputs=[cockpit_answer_output, cockpit_tts_checkbox],
            outputs=None,
            js=r"""
            (text, enabled) => {
                if (!enabled || !text || !('speechSynthesis' in window)) { return []; }
                const spoken = text
                    .replace(/[*_`#]/g, '')
                    .replace(/ft\/min/g, 'feet per minute')
                    .replace(/\bkt\b/g, 'knots')
                    .replace(/\bft\b/g, 'feet')
                    .replace(/°/g, ' degrees')
                    .replace(/\+(\d)/g, 'plus $1')
                    .replace(/(^|\s)-(\d)/g, '$1minus $2');
                window.speechSynthesis.cancel();   // don't queue answers back-to-back
                window.speechSynthesis.speak(new SpeechSynthesisUtterance(spoken));
                return [];
            }
            """,
        )
        cockpit_stop_speech_btn.click(
            fn=None, inputs=None, outputs=None,
            js="() => { if ('speechSynthesis' in window) { window.speechSynthesis.cancel(); } return []; }",
        )

    # ---------------- Radar panel ----------------
    with gr.Column(visible=False) as radar_view:
        radar_back_btn = gr.Button("← Back")
        with gr.Row():
            with gr.Column(scale=1):
                radar_image_input = gr.Image(type="filepath", label="Upload radar image",
                                              sources=["upload", "clipboard"])
                radar_preset_dropdown = gr.Dropdown(
                    choices=RADAR_PRESET_QUESTIONS, label="Sample questions", value=None)
                radar_question_box = gr.Textbox(
                    label="Question", placeholder="Ask a question about the aircraft on the radar...",
                    lines=2)
                radar_run_btn = gr.Button("Run Analysis", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("**Analysis Result**")
                radar_answer_output = gr.Markdown(elem_id="answer_box")
                radar_highlight_output = gr.Image(label="Highlighted result (locate/find queries only)")

        radar_preset_dropdown.change(fn=lambda q: q, inputs=radar_preset_dropdown,
                                      outputs=radar_question_box)
        radar_run_btn.click(
            fn=radar_run_analysis,
            inputs=[radar_image_input, radar_question_box],
            outputs=[radar_answer_output, radar_highlight_output],
        )

    # ---------------- View switching (landing <-> cockpit <-> radar) ----------------
    def show_cockpit():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    def show_radar():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    def show_landing():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    cockpit_card_btn.click(fn=show_cockpit, outputs=[landing_view, cockpit_view, radar_view])
    radar_card_btn.click(fn=show_radar, outputs=[landing_view, cockpit_view, radar_view])
    cockpit_back_btn.click(fn=show_landing, outputs=[landing_view, cockpit_view, radar_view])
    radar_back_btn.click(fn=show_landing, outputs=[landing_view, cockpit_view, radar_view])

if __name__ == "__main__":
    demo.launch(share=args.share)
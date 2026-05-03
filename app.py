import torch
import re
import time
import json
from threading import Thread
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

app = Flask(__name__,
            template_folder='interface/templates',
            static_folder='interface/static')

MODEL_PATH = "saved_model/model_final"
TOKENIZER_PATH = "saved_model/tokenizer"
DEVICE = "cuda:0"

PROMPT_HEAD = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.padding_side = "left"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE
)
model.eval()
print("Model ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    prompt = PROMPT_HEAD + question + PROMPT_TAIL
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.7,
        top_k=32,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    def event_stream():
        t_start = time.perf_counter()
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        full_response = ""
        for token_text in streamer:
            full_response += token_text
            yield f"data: {json.dumps({'token': token_text})}\n\n"

        thread.join()
        t_elapsed = time.perf_counter() - t_start
        num_tokens = len(tokenizer.encode(full_response, add_special_tokens=False))

        matches = re.findall(r"\\boxed\{([^}]*)\}", full_response)
        answer = matches[-1].strip() if matches else None

        yield f"data: {json.dumps({'done': True, 'answer': answer, 'elapsed_sec': round(t_elapsed, 2), 'num_tokens': num_tokens, 'tokens_per_sec': round(num_tokens / t_elapsed, 1) if t_elapsed > 0 else 0})}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

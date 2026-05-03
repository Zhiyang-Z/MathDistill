const form = document.getElementById("inputForm");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const chatArea = document.getElementById("chatArea");

let welcomeRemoved = false;

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function appendUserMessage(question) {
  const div = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-label">You</div>
    <div class="bubble bubble-user">${escapeHtml(question)}</div>
  `;
  chatArea.appendChild(div);
}

// Creates the assistant bubble with an empty reasoning div and returns refs to inner elements
function createAssistantBubble() {
  const div = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-label">MathDistill</div>
    <div class="bubble bubble-assistant">
      <div class="reasoning streaming"></div>
    </div>
  `;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;

  const bubble = div.querySelector(".bubble-assistant");
  const reasoning = div.querySelector(".reasoning");
  return { bubble, reasoning };
}

function appendLoading() {
  const div = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-label">MathDistill</div>
    <div class="loading-bubble">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
  `;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

function finalizeResponse(bubble, reasoning, answer, elapsed, numTokens, tokensPerSec) {
  reasoning.classList.remove("streaming");

  if (answer !== null && answer !== undefined) {
    const answerBox = document.createElement("div");
    answerBox.className = "answer-box";
    answerBox.innerHTML = `
      <div class="answer-label">Final Answer</div>
      <div class="answer-value">\\(${escapeHtml(answer)}\\)</div>
    `;
    bubble.appendChild(answerBox);
  }

  const statsBar = document.createElement("div");
  statsBar.className = "stats-bar";
  statsBar.innerHTML = `
    <span class="stat"><span class="stat-icon">&#x23F1;</span> ${elapsed}s</span>
    <span class="stat-sep">·</span>
    <span class="stat"><span class="stat-icon">&#x2261;</span> ${numTokens} tokens</span>
    <span class="stat-sep">·</span>
    <span class="stat"><span class="stat-icon">&#x26A1;</span> ${tokensPerSec} tok/s</span>
  `;
  bubble.appendChild(statsBar);

  if (window.MathJax) {
    MathJax.typesetPromise([bubble]).catch(console.error);
  }
}

function showError(message) {
  const div = document.createElement("div");
  div.className = "message";
  div.innerHTML = `
    <div class="msg-label">MathDistill</div>
    <div class="error-bubble">Error: ${escapeHtml(message)}</div>
  `;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function submitQuestion(question) {
  if (!welcomeRemoved) {
    const welcome = chatArea.querySelector(".welcome");
    if (welcome) welcome.remove();
    welcomeRemoved = true;
  }

  appendUserMessage(question);

  // Show 3-dot loader while waiting for the first token
  const loadingEl = appendLoading();
  sendBtn.disabled = true;
  input.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const data = await res.json();
      loadingEl.remove();
      showError(data.error || "Server error");
      return;
    }

    // Switch from loader to streaming bubble on first byte
    let bubbleCreated = false;
    let bubble, reasoning;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // last part may be incomplete

      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        let evt;
        try { evt = JSON.parse(part.slice(6)); } catch { continue; }

        if (evt.done) {
          finalizeResponse(bubble, reasoning, evt.answer, evt.elapsed_sec, evt.num_tokens, evt.tokens_per_sec);
        } else {
          if (!bubbleCreated) {
            loadingEl.remove();
            ({ bubble, reasoning } = createAssistantBubble());
            bubbleCreated = true;
          }
          reasoning.textContent += evt.token;
          chatArea.scrollTop = chatArea.scrollHeight;
        }
      }
    }
  } catch (err) {
    loadingEl.remove();
    showError("Could not reach server.");
  } finally {
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
    chatArea.scrollTop = chatArea.scrollHeight;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  submitQuestion(question);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event("submit"));
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});

function fillExample(btn) {
  input.value = btn.textContent.trim();
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
  input.focus();
}

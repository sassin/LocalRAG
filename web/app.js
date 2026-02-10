const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const providerEl = document.getElementById("provider");
const composerEl = document.getElementById("composer");
const SESSION_KEY = "localrag_session_id";
const KEY_STORAGE = "localrag_access_key";

function getAccessKey() {
  return localStorage.getItem(KEY_STORAGE) || "";
}
function setAccessKey(k) {
  localStorage.setItem(KEY_STORAGE, k);
}
function promptForKey(force=false) {
  if (!force && getAccessKey()) return;
  const k = window.prompt("Enter access key:");
  if (k && k.trim()) setAccessKey(k.trim());
}

promptForKey(false);

// ----- Chat UX helpers -----
function isNearBottom(px = 120) {
  const distance = chatEl.scrollHeight - (chatEl.scrollTop + chatEl.clientHeight);
  return distance < px;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMsg(text, who, provider=null) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;

  if (who === "agent" && provider) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `provider: ${provider}`;
    div.appendChild(meta);
  }

  const keepPinned = isNearBottom();
  chatEl.appendChild(div);
  if (keepPinned) scrollToBottom();
}

// ----- Keyboard + layout management (WhatsApp-style) -----

function setComposerHeight() {
  const rect = composerEl.getBoundingClientRect();
  // include wrap padding (~20px)
  const wrapPadding = 28;
  const h = Math.ceil(rect.height + wrapPadding);
  document.documentElement.style.setProperty("--composer-h", `${h}px`);
}

function setKeyboardHeight() {
  // The “real” keyboard height is the part of the layout viewport not covered by the visual viewport.
  // On iOS/Android, visualViewport.height shrinks when keyboard opens.
  const vv = window.visualViewport;
  if (!vv) {
    document.documentElement.style.setProperty("--kb", "0px");
    return;
  }

  const kb = Math.max(0, Math.round(window.innerHeight - vv.height - vv.offsetTop));
  document.documentElement.style.setProperty("--kb", `${kb}px`);
}

function autosizeTextarea() {
  inputEl.style.height = "0px";
  const newH = Math.min(inputEl.scrollHeight, 180);
  inputEl.style.height = `${Math.max(newH, 44)}px`;
  setComposerHeight();
}

let rafId = null;
let settleTimer = null;

function measureAndApplyLayout({ settle = false } = {}) {
  const pinned = isNearBottom();

  setKeyboardHeight();
  setComposerHeight();

  document.documentElement.style.setProperty(
  "--kb",
  (isNearBottom() ? document.documentElement.style.getPropertyValue("--kb") : "0px")
  );

  // Only keep pinned users pinned
  if (pinned) {
    // During animation, don't fight it—scroll after measurements
    scrollToBottom();
  }

  // One final settle after keyboard finishes animating
  if (settleTimer) clearTimeout(settleTimer);
  if (settle) {
    settleTimer = setTimeout(() => {
      const pinned2 = isNearBottom();
      setKeyboardHeight();
      setComposerHeight();
      if (pinned2) scrollToBottom();
    }, 220); // ✅ avoids mid-animation bounce (works well across devices)
  }
}

function getSessionId() {
  let sid = localStorage.getItem(SESSION_KEY);
  if (!sid) {
    sid = crypto.randomUUID().replaceAll("-", "");
    localStorage.setItem(SESSION_KEY, sid);
  }
  return sid;
}

async function sendMessage(message, provider) {
  const session_id = getSessionId();

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      provider,
      session_id, // ✅ key change
      mode: "rag",
    }),
  });

  const data = await res.json();

  // If server ever rotates/creates a session_id, keep it
  if (data.session_id && data.session_id !== session_id) {
    localStorage.setItem(SESSION_KEY, data.session_id);
  }

  return data;
}

function scheduleLayoutUpdate({ settle = false } = {}) {
  if (rafId) return;
  rafId = requestAnimationFrame(() => {
    rafId = null;
    measureAndApplyLayout({ settle });
  });
}

function onViewportChange() {
  scheduleLayoutUpdate({ settle: false });
}

// Listeners
window.addEventListener("resize", onViewportChange);

if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", onViewportChange);
  window.visualViewport.addEventListener("scroll", onViewportChange);
}

// On focus: allow keyboard to animate, then settle once
inputEl.addEventListener("focus", () => {
  scheduleLayoutUpdate({ settle: true });
});

// Optional: on blur, settle back
inputEl.addEventListener("blur", () => {
  scheduleLayoutUpdate({ settle: true });
});


inputEl.addEventListener("input", autosizeTextarea);

// Initial measurements
setTimeout(() => {
  autosizeTextarea();
  handleViewportChange();
}, 0);

// ----- Send logic -----
async function send() {
  const q = inputEl.value.trim();
  if (!q) return;

  addMsg(q, "user");
  inputEl.value = "";
  autosizeTextarea();
  sendBtn.disabled = true;

  const provider = providerEl.value;
  let accessKey = getAccessKey();

  try {
    let res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(accessKey ? { "X-ACCESS-KEY": accessKey } : {}),
      },
      body: JSON.stringify({ message: q, provider }),
    });

    if (res.status === 401) {
      promptForKey(true);
      accessKey = getAccessKey();
      res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(accessKey ? { "X-ACCESS-KEY": accessKey } : {}),
        },
        body: JSON.stringify({ message: q, provider }),
      });
    }

    const data = await res.json();
    if (!res.ok) {
      addMsg(`Error: ${data.detail || res.statusText}`, "agent", provider);
    } else {
      addMsg(data.answer || "(no response)", "agent", data.provider);
    }
  } catch (e) {
    addMsg(`Network error: ${e}`, "agent", provider);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", send);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

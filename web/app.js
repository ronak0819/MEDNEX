const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#chat-messages");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-chat");
const chatModeButton = document.querySelector("#chat-mode");
const agentModeButton = document.querySelector("#agent-mode");
const examplesPanel = document.querySelector(".examples");
const assistantName = document.querySelector("#assistant-name");
const assistantStatus = document.querySelector("#assistant-status");
const messageLabel = document.querySelector("#message-label");

let activeMode = "chat";
let agentSessionId = null;
let awaitingAgentAnswer = false;

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function addUserMessage(text) {
  const article = document.createElement("article");
  article.className = "message user-message";
  const content = document.createElement("div");
  content.className = "message-content";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "You";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  content.append(label, bubble);
  article.append(content);
  messages.append(article);
  scrollToLatest();
}

function addAssistantReply(text) {
  const article = document.createElement("article");
  article.className = "message assistant-message";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "M";
  const content = document.createElement("div");
  content.className = "message-content";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = activeMode === "agent"
    ? "MEDNEX · educational information"
    : "MEDNEX · general information";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  content.append(label, bubble);
  article.append(avatar, content);
  messages.append(article);
  scrollToLatest();
}

function addError(message) {
  addAssistantReply(
    `${message}\n\nPlease try again. No message is stored by this local demonstration.`,
  );
}

function addTypingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant-message typing";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "M";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = activeMode === "agent"
    ? "Preparing a plain-English educational response…"
    : "Putting together general information…";
  article.append(avatar, bubble);
  messages.append(article);
  scrollToLatest();
  return article;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "The local MEDNEX service could not process that request.");
  }
  return data;
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || sendButton.disabled) return;
  addUserMessage(trimmed);
  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;
  const typing = addTypingMessage();

  try {
    if (activeMode === "chat") {
      const payload = await postJson("/api/chat", { message: trimmed });
      typing.remove();
      addAssistantReply(payload.response);
    } else if (agentSessionId && awaitingAgentAnswer) {
      const payload = await postJson("/api/agent/answer", {
        session_id: agentSessionId,
        answer: trimmed,
      });
      typing.remove();
      showAgentAnswer(payload);
    } else {
      const payload = await postJson("/api/agent/start", { description: trimmed });
      typing.remove();
      showAgentAnswer(payload);
    }
  } catch (error) {
    typing.remove();
    addError(error instanceof Error ? error.message : "The request failed.");
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

function showAgentAnswer(payload) {
  agentSessionId = payload.session_id;
  awaitingAgentAnswer = payload.awaiting_answer;
  assistantStatus.textContent = awaitingAgentAnswer
    ? "One follow-up question"
    : "Educational information · no diagnosis";
  input.placeholder = awaitingAgentAnswer
    ? "Type your answer to the question…"
    : "Start another fictional case…";
  addAssistantReply(payload.answer);
}

function showWelcome(mode) {
  messages.replaceChildren();
  const article = document.createElement("article");
  article.className = "message assistant-message";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "M";
  const content = document.createElement("div");
  content.className = "message-content";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = mode === "agent"
    ? "MEDNEX · educational information"
    : "MEDNEX · general information";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble welcome-bubble";
  bubble.textContent = mode === "agent"
    ? "Describe a fictional or synthetic case in ordinary English. I’ll share a simple educational summary and may ask one follow-up question. Do not enter real patient information."
    : "Hello. Tell me what you’re experiencing, and I’ll share general comfort measures and signs that mean it’s time to seek help. I can’t diagnose or recommend medicines.";
  content.append(label, bubble);
  article.append(avatar, content);
  messages.append(article);
  scrollToLatest();
}

function setMode(mode) {
  activeMode = mode;
  examplesPanel.hidden = mode === "agent";
  agentSessionId = null;
  awaitingAgentAnswer = false;
  chatModeButton.classList.toggle("active", mode === "chat");
  agentModeButton.classList.toggle("active", mode === "agent");
  chatModeButton.setAttribute("aria-selected", String(mode === "chat"));
  agentModeButton.setAttribute("aria-selected", String(mode === "agent"));
  assistantName.textContent = mode === "agent"
    ? "MEDNEX Reasoning Agent"
    : "MEDNEX assistant";
  assistantStatus.textContent = mode === "agent"
    ? "Plain-English educational summary · no LLM"
    : "General information · no LLM";
  messageLabel.textContent = mode === "agent"
    ? "Describe a fictional or synthetic patient case"
    : "Describe a symptom or ask a general question";
  input.placeholder = mode === "agent"
    ? "Example: A fictional patient has cough and fever…"
    : "What would you like general information about?";
  showWelcome(mode);
  input.focus();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt || ""));
});

chatModeButton.addEventListener("click", () => setMode("chat"));
agentModeButton.addEventListener("click", () => setMode("agent"));

clearButton.addEventListener("click", () => {
  agentSessionId = null;
  awaitingAgentAnswer = false;
  input.value = "";
  input.placeholder = activeMode === "agent"
    ? "Example: A fictional patient has cough and fever…"
    : "What would you like general information about?";
  showWelcome(activeMode);
  input.focus();
});

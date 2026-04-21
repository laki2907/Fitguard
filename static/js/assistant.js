/* ── FitGuard Assistant Sidebar ─────────────────────────────────────────── */
(function () {
  "use strict";

  const fab      = document.getElementById("fg-fab");
  const sidebar  = document.getElementById("fg-sidebar");
  const closeBtn = document.getElementById("fg-close");
  const form     = document.getElementById("fg-form");
  const input    = document.getElementById("fg-input");
  const messages = document.getElementById("fg-messages");
  const quickBtns = document.querySelectorAll(".fg-quick");

  if (!fab || !sidebar) return; // not logged in or elements missing

  // ── Open / close ──────────────────────────────────────────────────────────
  function openSidebar() {
    sidebar.classList.add("fg-open");
    fab.classList.add("fg-fab-hidden");
    input.focus();
  }

  function closeSidebar() {
    sidebar.classList.remove("fg-open");
    fab.classList.remove("fg-fab-hidden");
  }

  fab.addEventListener("click", openSidebar);
  closeBtn.addEventListener("click", closeSidebar);

  // Close on Escape key
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && sidebar.classList.contains("fg-open")) closeSidebar();
  });

  // ── Markdown-lite renderer ────────────────────────────────────────────────
  // Handles **bold** and bullet lists (lines starting with •)
  function renderMarkdown(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
  }

  // ── Append a message bubble ───────────────────────────────────────────────
  function appendMsg(text, role) {
    const row    = document.createElement("div");
    row.className = `fg-msg fg-msg-${role}`;
    const bubble = document.createElement("div");
    bubble.className = "fg-bubble";
    bubble.innerHTML  = role === "bot" ? renderMarkdown(text) : escapeHtml(text);
    row.appendChild(bubble);
    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
    return row;
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, c =>
      ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])
    );
  }

  // ── Typing indicator ──────────────────────────────────────────────────────
  function showTyping() {
    const row    = document.createElement("div");
    row.className = "fg-msg fg-msg-bot fg-typing";
    row.id        = "fg-typing-indicator";
    row.innerHTML = '<div class="fg-bubble"><span class="fg-dot"></span><span class="fg-dot"></span><span class="fg-dot"></span></div>';
    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
  }

  function hideTyping() {
    const el = document.getElementById("fg-typing-indicator");
    if (el) el.remove();
  }

  // ── Send message to /assistant ────────────────────────────────────────────
  async function send(text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    appendMsg(trimmed, "user");
    input.value = "";
    input.disabled = true;

    showTyping();

    try {
      const res  = await fetch("/assistant", {
        method : "POST",
        headers: { "Content-Type": "application/json" },
        body   : JSON.stringify({ message: trimmed }),
      });

      hideTyping();

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      appendMsg(data.reply || "Sorry, I couldn't get a response.", "bot");
    } catch (err) {
      hideTyping();
      appendMsg("⚠️ Couldn't reach the assistant. Are you still logged in?", "bot");
      console.error("FitGuard Assistant error:", err);
    } finally {
      input.disabled = false;
      input.focus();
    }
  }

  // ── Form submit ───────────────────────────────────────────────────────────
  form.addEventListener("submit", e => {
    e.preventDefault();
    send(input.value);
  });

  // ── Quick buttons ─────────────────────────────────────────────────────────
  quickBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      send(btn.dataset.msg);
    });
  });

})();
/* ────────────────────────────────────────────────────────────────────────── */
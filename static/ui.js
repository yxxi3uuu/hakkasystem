(function () {
  const paths = {
    home: '<path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
    camera: '<path d="M14.5 4 16 7h3a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h3l1.5-3z"/><circle cx="12" cy="13" r="3.5"/>',
    mic: '<rect x="9" y="2.5" width="6" height="12" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M9 21h6"/>',
    practice: '<circle cx="12" cy="12" r="8"/><path d="m9.5 8.5 6 3.5-6 3.5z"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    book: '<path d="M3 5a3 3 0 0 1 3-2h5v17H6a3 3 0 0 0-3 2zM21 5a3 3 0 0 0-3-2h-5v17h5a3 3 0 0 1 3 2z"/>',
    image: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9" r="1.5"/><path d="m21 15-5-5L5 20"/>',
    game: '<path d="M8 7h8a5 5 0 0 1 4.7 6.7l-1.2 3.4a2.4 2.4 0 0 1-4 1l-1.4-1.6H9.9l-1.4 1.6a2.4 2.4 0 0 1-4-1l-1.2-3.4A5 5 0 0 1 8 7z"/><path d="M7 12h4M9 10v4M16 11h.01M18 13h.01"/>',
    lightbulb: '<path d="M9 18h6M10 22h4"/><path d="M8.3 15.5a7 7 0 1 1 7.4 0c-.5.4-.7 1-.7 1.5H9c0-.5-.2-1.1-.7-1.5z"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    close: '<path d="M18 6 6 18M6 6l12 12"/>',
    save: '<path d="M5 3h12l3 3v15H4V3z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/>',
    volume: '<path d="M11 5 6 9H3v6h3l5 4zM15 9a4 4 0 0 1 0 6M18 6a8 8 0 0 1 0 12"/>',
    bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/>',
    edit: '<path d="m4 20 4.5-1L19 8.5 15.5 5 5 15.5zM13.5 7l3.5 3.5"/>',
    upload: '<path d="M12 16V3M7 8l5-5 5 5M4 15v6h16v-6"/>',
    trash: '<path d="M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14M10 11v6M14 11v6"/>',
    refresh: '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M18.5 9A7 7 0 0 0 6 6.5L4 9M5.5 15A7 7 0 0 0 18 17.5l2-2.5"/>',
    undo: '<path d="m9 7-5 5 5 5"/><path d="M4 12h10a6 6 0 0 1 6 6v1"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H3v-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.6V3h4v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1z"/>',
    star: '<path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z"/>',
    award: '<circle cx="12" cy="9" r="6"/><path d="m8 14-1 7 5-3 5 3-1-7"/>',
    alert: '<path d="M12 3 2 21h20z"/><path d="M12 9v5M12 18h.01"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/>',
    folder: '<path d="M3 6h7l2 2h9v11H3z"/>',
    lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    send: '<path d="m22 2-7 20-4-9-9-4zM22 2 11 13"/>',
    logout: '<path d="M10 4H4v16h6M14 8l4 4-4 4M8 12h10"/>',
    sparkle: '<path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4zM18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>'
  };

  function icon(name, extraClass) {
    return `<svg class="app-icon${extraClass ? ` ${extraClass}` : ""}" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">${paths[name] || paths.sparkle}</svg>`;
  }

  const emojiToIcon = new Map(Object.entries({
    "📷":"camera", "📸":"camera", "🎤":"mic", "🎙":"mic", "🎮":"game", "🧩":"game",
    "📊":"chart", "📈":"chart", "👤":"user", "👥":"user", "📚":"book", "📖":"book",
    "🖼":"image", "💡":"lightbulb", "✅":"check", "✓":"check", "❌":"close", "✕":"close",
    "💾":"save", "🔊":"volume", "🔔":"bell", "✏":"edit", "📝":"edit", "📤":"send",
    "🗑":"trash", "🔄":"refresh", "↩":"undo", "🎯":"target", "⚙":"settings", "⭐":"star", "🌟":"star",
    "🏆":"award", "⚠":"alert", "🚫":"alert", "📁":"folder", "📂":"folder", "🔑":"lock",
    "🛡":"lock", "🚪":"logout", "🏠":"home", "🔥":"award", "🎓":"award", "🔍":"target",
    "📋":"edit", "📄":"edit", "📣":"bell", "📢":"bell", "➕":"sparkle", "⚡":"sparkle",
    "🧠":"sparkle", "🤖":"sparkle", "💪":"award", "🎉":"award", "🀄":"game", "🃏":"game",
    "🔗":"game", "🔲":"target", "🏷":"target", "🗣":"volume"
  }));
  const emojiPattern = /(?:[\u{1F000}-\u{1FAFF}\u2600-\u27BF]|\u21A9)\uFE0F?/gu;

  function replaceEmojiText(root) {
    if (!root || root.nodeType !== Node.ELEMENT_NODE) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((node) => {
      if (!emojiPattern.test(node.nodeValue)) return;
      emojiPattern.lastIndex = 0;
      const parent = node.parentElement;
      if (!parent || parent.closest("script,style,textarea,svg")) return;
      if (parent.matches("option,title")) {
        node.nodeValue = node.nodeValue.replace(emojiPattern, "").replace(/\s{2,}/g, " ");
        return;
      }
      const fragment = document.createDocumentFragment();
      let last = 0;
      for (const match of node.nodeValue.matchAll(emojiPattern)) {
        if (match.index > last) fragment.append(node.nodeValue.slice(last, match.index));
        const holder = document.createElement("span");
        holder.className = "inline-icon";
        holder.innerHTML = icon(emojiToIcon.get(match[0].replace("\uFE0F", "")) || "sparkle");
        fragment.append(holder);
        last = match.index + match[0].length;
      }
      if (last < node.nodeValue.length) fragment.append(node.nodeValue.slice(last));
      node.replaceWith(fragment);
    });
  }

  const clickableSelector = [
    "a[href]", "button", "[onclick]", "[role='button']", "summary", "label[for]",
    "input[type='button']", "input[type='submit']", "input[type='reset']",
    ".word-card", ".vocab-item", ".menu-item", ".notif-item",
    ".quick-card", ".game-card", ".game-tab", ".match-item", ".flip-card",
    ".listen-opt", ".mode-tab", ".src-tab", "[id^='recCard']"
  ].join(",");

  function markClickables(root) {
    if (!root || root.nodeType !== Node.ELEMENT_NODE) return;
    if (root.matches(clickableSelector)) root.classList.add("app-clickable");
    root.querySelectorAll(clickableSelector).forEach((element) => {
      element.classList.add("app-clickable");
    });
  }

  const nav = document.querySelector(".bottom-nav");
  if (nav) {
    const path = window.location.pathname.replace(/\/+$/, "") || "/";
    const items = [
      { href: "/", icon: "home", label: "首頁", matches: ["/"] },
      { href: "/practice", icon: "practice", label: "練習", matches: ["/practice", "/game"] },
      { href: "/recognition", icon: "camera", label: "辨識", matches: ["/recognition"], special: true },
      { href: "/record", icon: "chart", label: "紀錄", matches: ["/record", "/learning"] },
      { href: "/profile", icon: "user", label: "我的", matches: ["/profile", "/admin"] },
    ];
    nav.innerHTML = items.map((item) => {
      const active = item.matches.includes(path);
      return `<a class="nav-item${active ? " active" : ""}${item.special ? " nav-recognition" : ""}"
        href="${item.href}" ${active ? 'aria-current="page"' : ""}>
        <span class="nav-icon">${icon(item.icon)}</span>
        <span class="nav-label">${item.label}</span>
      </a>`;
    }).join("");
    nav.setAttribute("aria-label", "主要導覽");
  }

  const attribution = document.createElement("footer");
  attribution.className = "api-attribution";
  attribution.setAttribute("role", "contentinfo");
  attribution.textContent = "本產品使用客家委員會授權／提供之 API";
  if (nav) {
    nav.insertAdjacentElement("beforebegin", attribution);
  } else {
    document.body.appendChild(attribution);
  }

  replaceEmojiText(document.body);
  markClickables(document.body);
  new MutationObserver((mutations) => {
    mutations.forEach((mutation) => mutation.addedNodes.forEach((node) => {
      if (node.nodeType === Node.ELEMENT_NODE) {
        replaceEmojiText(node);
        markClickables(node);
      }
      if (node.nodeType === Node.TEXT_NODE && node.parentElement) replaceEmojiText(node.parentElement);
    }));
  }).observe(document.body, { childList: true, subtree: true });

  const nativeAlert = window.alert.bind(window);
  window.alert = (message) => nativeAlert(String(message).replace(emojiPattern, "").trim());

  document.querySelectorAll("button:not([aria-label])").forEach((button) => {
    const text = button.textContent.trim().replace(/\s+/g, " ");
    if (text) button.setAttribute("aria-label", text);
  });
  document.querySelectorAll("img:not([loading])").forEach((image) => { image.loading = "lazy"; });

  // 手機整頁左右滑動導覽：左滑下一頁、右滑上一頁。
  // 互動元件、相機、遊戲、彈窗與螢幕邊緣不攔截，避免誤觸。
  if (nav) {
    const pages = ["/", "/practice", "/recognition", "/record", "/profile"];
    const aliases = {
      "/game": "/practice",
      "/learning": "/record",
      "/admin": "/profile"
    };
    const currentPath = window.location.pathname.replace(/\/+$/, "") || "/";
    const currentPage = aliases[currentPath] || currentPath;
    const excludedSelector = [
      "a", "button", "input", "select", "textarea", "label",
      "[onclick]", "[contenteditable='true']", ".mode-tabs", ".game-tabs", ".src-tabs",
      ".camera-wrap", ".camera-controls", "#gameArea", ".modal-overlay",
      ".word-modal-overlay", ".confirm-overlay", ".notif-overlay", ".bottom-nav"
    ].join(",");
    let swipeStart = null;

    document.addEventListener("pointerdown", (event) => {
      if (event.pointerType !== "touch" || !event.isPrimary) return;
      if (event.clientX < 24 || event.clientX > window.innerWidth - 24) return;
      if (event.target.closest(excludedSelector)) return;
      swipeStart = { x: event.clientX, y: event.clientY, time: Date.now(), id: event.pointerId };
    }, { passive: true });

    document.addEventListener("pointerup", (event) => {
      if (!swipeStart || event.pointerId !== swipeStart.id) return;
      const start = swipeStart;
      swipeStart = null;
      const dx = event.clientX - start.x;
      const dy = event.clientY - start.y;
      const elapsed = Date.now() - start.time;
      if (elapsed > 700 || Math.abs(dx) < 72 || Math.abs(dx) < Math.abs(dy) * 1.35) return;

      const index = pages.indexOf(currentPage);
      if (index < 0) return;
      const nextIndex = dx < 0 ? index + 1 : index - 1;
      if (nextIndex < 0 || nextIndex >= pages.length) return;

      document.body.classList.add(dx < 0 ? "page-swipe-left" : "page-swipe-right");
      window.setTimeout(() => { window.location.href = pages[nextIndex]; }, 120);
    }, { passive: true });

    document.addEventListener("pointercancel", () => { swipeStart = null; }, { passive: true });
  }
})();

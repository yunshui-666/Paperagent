"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  sessions: [],
  collections: [],
  documents: [],
  turns: [],
  sessionId: null,
  collection: "",
  documentIds: [],
  options: {
    mode: "hybrid_rerank",
    dense_k: 20,
    keyword_k: 20,
    top_k: 10,
    rewrite_query: true,
  },
  busy: new Set(),
  creating: false,
};
let toastTimer,
  renameId,
  refreshRunning = false;
const stages = {
  queued: "排队中",
  starting: "准备处理",
  parsing: "解析中",
  checking_cache: "核对缓存",
  chunking: "切块中",
  embedding: "向量化中",
  saving: "保存中",
  completed: "已入库",
  failed: "处理失败",
};
const labels = {
  paper_chunks: "原有知识库",
  paper_chunks1: "本地文本解析",
  paper_chunks2: "本地 OCR",
  paper_chunks3: "本地 OCR + VLM 双路",
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function button(text, className, action) {
  const element = node("button", className, text);
  element.type = "button";
  element.addEventListener("click", action);
  return element;
}
function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("toast").classList.add("hidden"), 4500);
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("服务返回格式异常，请查看服务器日志");
  }
  if (!response.ok) {
    let detail =
      data.detail ||
      data.items?.find((item) => item.error)?.error.message ||
      `请求失败（${response.status}）`;
    if (Array.isArray(detail))
      detail = detail.map((item) => item.msg).join("；");
    throw new Error(detail);
  }
  return data;
}
function jsonOptions(method, body) {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
function collectionReady() {
  const collection = state.collections.find(
    (item) => item.name === state.collection,
  );
  return Boolean(
    collection && (collection.ready_chunk_count ?? collection.chunk_count) > 0,
  );
}
function updateComposer() {
  const busy = state.busy.has(state.sessionId) || state.creating;
  $("send").disabled = busy || !collectionReady();
  $("send").textContent = busy ? "回答中…" : "发送 ↑";
  $("question").disabled = busy || !collectionReady();
  $("composer-hint").textContent = !collectionReady()
    ? "先完成论文入库，再开始提问"
    : state.documentIds.length
      ? `本次研读 ${state.documentIds.length} 篇论文`
      : "回答附带可回查的论文证据";
  $("scope-button").textContent = state.documentIds.length
    ? `${state.documentIds.length} 篇论文`
    : "全部论文";
}
function renderSessions() {
  const root = $("sessions");
  root.replaceChildren();
  for (const session of state.sessions) {
    const row = node(
      "div",
      "session-row" + (session.id === state.sessionId ? " active" : ""),
    );
    row.append(
      button(session.title, "session-main", () =>
        loadSession(session.id).catch((error) => toast(error.message)),
      ),
    );
    const actions = node("div", "session-tools");
    const rename = button("改名", "", () => {
      renameId = session.id;
      $("session-title").value = session.title;
      $("rename-dialog").showModal();
    });
    rename.setAttribute("aria-label", `重命名 ${session.title}`);
    const remove = button("×", "", async () => {
      if (
        !(await confirmAction(
          "删除会话",
          `将删除“${session.title}”及对应会话历史。`,
        ))
      )
        return;
      try {
        await api(`/sessions/${session.id}`, { method: "DELETE" });
        if (state.sessionId === session.id) newChat();
        await refresh();
      } catch (error) {
        toast(error.message);
      }
    });
    remove.setAttribute("aria-label", `删除 ${session.title}`);
    actions.append(rename, remove);
    row.append(actions);
    root.append(row);
  }
}
function newChat() {
  state.sessionId = null;
  state.turns = [];
  localStorage.removeItem("paperagent-session");
  renderSessions();
  renderMessages();
  updateComposer();
  if (collectionReady()) $("question").focus();
}
async function loadSession(id) {
  const session = await api(`/sessions/${id}`);
  state.sessionId = id;
  state.turns = session.turns;
  state.collection = session.collection_name;
  const { document_ids, ...options } = session.options;
  state.options = options;
  state.documentIds = [...document_ids];
  $("collection").value = state.collection;
  localStorage.setItem("paperagent-session", id);
  renderSessions();
  renderMessages();
  updateComposer();
}
function addCitationButtons(root, sources) {
  const byId = new Map(sources.map((source) => [String(source.id), source]));
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const targets = [];
  while (walker.nextNode()) {
    const text = walker.currentNode;
    if (
      !text.parentElement.closest("pre,code,a,button") &&
      /\[\d+\]/.test(text.textContent)
    )
      targets.push(text);
  }
  for (const text of targets) {
    const fragment = document.createDocumentFragment();
    let end = 0;
    for (const match of text.textContent.matchAll(/\[(\d+)\]/g)) {
      fragment.append(
        document.createTextNode(text.textContent.slice(end, match.index)),
      );
      const source = byId.get(match[1]);
      fragment.append(
        source
          ? button(match[0], "inline-citation", () => showEvidence(source))
          : document.createTextNode(match[0]),
      );
      end = match.index + match[0].length;
    }
    fragment.append(document.createTextNode(text.textContent.slice(end)));
    text.replaceWith(fragment);
  }
  for (const anchor of root.querySelectorAll("a")) {
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
  }
}
function renderMessages() {
  const root = $("messages");
  const atBottom = root.scrollHeight - root.scrollTop - root.clientHeight < 160;
  root.replaceChildren();
  if (!state.turns.length) {
    const empty = node("div", "empty-state");
    empty.append(
      node("div", "empty-emblem", "P"),
      node("h1", "", "让论文成为\n可追问的知识"),
      node(
        "p",
        "",
        collectionReady()
          ? "从研究问题、方法和实验结果开始，沿着引用回到原文，也可以继续追问。"
          : "上传论文后，解析与入库会在后台完成。你可以在文档库查看进度。",
      ),
    );
    empty.querySelector("h1").style.whiteSpace = "pre-line";
    if (collectionReady()) {
      const suggestions = node("div", "suggestions");
      for (const [title, question, hint] of [
        [
          "理解研究方法",
          "请概括论文的研究问题、核心方法和主要贡献。",
          "研究问题与方法",
        ],
        [
          "查看实验证据",
          "论文的实验结果支持了哪些结论？请注明证据。",
          "结果与证据",
        ],
        ["梳理适用边界", "论文方法有哪些局限和适用条件？", "局限与条件"],
      ]) {
        const item = button(title, "suggestion", () => sendQuestion(question));
        item.append(node("span", "", hint));
        suggestions.append(item);
      }
      empty.append(suggestions);
    } else empty.append(button("打开文档库", "button primary", openLibrary));
    root.append(empty);
    return;
  }
  for (const turn of state.turns) {
    const article = node("article", "turn");
    article.append(
      node("div", "question-bubble", turn.question),
      node("div", "answer-label", "PaperAgent"),
    );
    if (turn.status === "completed") {
      const content = node("div", "answer-content");
      // answer_html 由服务器 MarkdownIt 在禁用原始 HTML 后生成。
      content.innerHTML = turn.answer_html || "";
      if (!turn.answer_html) content.textContent = turn.answer || "";
      addCitationButtons(content, turn.sources || []);
      article.append(content);
      const sources = node("div", "sources-row");
      for (const source of turn.sources || [])
        sources.append(
          button(
            `[${source.id}] ${source.source}${source.page ? ` · 第 ${source.page} 页` : ""}`,
            "source-chip",
            () => showEvidence(source),
          ),
        );
      article.append(sources);
      const footer = node("div", "turn-footer");
      footer.append(
        button("复制回答", "", async () => {
          try {
            await navigator.clipboard.writeText(turn.answer);
            toast("已复制回答");
          } catch {
            toast("复制失败，可以选中文本复制");
          }
        }),
      );
      const metrics = turn.metrics || {};
      if (metrics.evidence_origin === "web") {
        footer.append(node("span", "", "联网搜索回答"));
      }
      if (metrics.evidence_origin === "papers") {
        footer.append(node("span", "", "论文搜索与入库"));
      }
      const total = metrics.request_seconds;
      if (total !== undefined)
        footer.append(
          node(
            "span",
            "",
            `${Number(total).toFixed(2)} 秒 · ${turn.sources?.length || 0} 条证据`,
          ),
        );
      const usage = metrics.generation_usage || {};
      if (usage.total_tokens)
        footer.append(node("span", "", `${usage.total_tokens} Token`));
      const details = node("details", "");
      details.append(node("summary", "", "查看检索与耗时"));
      details.append(node("pre", "", JSON.stringify(metrics, null, 2)));
      footer.append(details);
      article.append(footer);
    } else if (turn.status === "failed") {
      const error = node(
        "div",
        "error-card",
        turn.error || "本次未完成，请重试。",
      );
      error.append(
        document.createElement("br"),
        button("重试问题", "button", () =>
          sendQuestion(turn.question, turn.request_id),
        ),
      );
      article.append(error);
    } else article.append(node("div", "pending", "正在检索论文并组织回答"));
    root.append(article);
  }
  if (atBottom || state.busy.has(state.sessionId))
    root.scrollTop = root.scrollHeight;
}
async function sendQuestion(question, requestId = null) {
  question = question.trim();
  if (!question || state.busy.has(state.sessionId) || state.creating) return;
  if (!collectionReady()) {
    openLibrary();
    return;
  }
  let id = state.sessionId;
  try {
    if (!id) {
      state.creating = true;
      updateComposer();
      const session = await api(
        "/sessions",
        jsonOptions("POST", {
          title: "新对话",
          collection_name: state.collection,
          document_ids: state.documentIds,
          ...state.options,
        }),
      );
      id = session.id;
      state.sessionId = id;
      localStorage.setItem("paperagent-session", id);
    }
  } catch (error) {
    toast(error.message);
    return;
  } finally {
    state.creating = false;
    updateComposer();
  }
  const request = requestId || crypto.randomUUID();
  state.busy.add(id);
  $("question").value = "";
  $("question").style.height = "auto";
  const existing = state.turns.find((turn) => turn.request_id === request);
  if (existing) {
    existing.status = "running";
    existing.error = null;
  } else
    state.turns.push({
      question,
      request_id: request,
      status: "running",
      sources: [],
    });
  renderMessages();
  updateComposer();
  try {
    await api(
      "/chat",
      jsonOptions("POST", { thread_id: id, request_id: request, question }),
    );
    if (state.sessionId === id) await loadSession(id);
    await refresh();
  } catch (error) {
    toast(error.message);
    if (state.sessionId === id) {
      try {
        await loadSession(id);
      } catch {
        const turn = state.turns.find((item) => item.request_id === request);
        if (turn) {
          turn.status = "failed";
          turn.error = error.message;
        }
        renderMessages();
      }
    }
  } finally {
    state.busy.delete(id);
    updateComposer();
    if (state.sessionId === id) $("question").focus();
  }
}
function showEvidence(source) {
  const isWeb = source.kind === "web";
  $("evidence-title").textContent = `[${source.id}] ${source.source}`;
  $("evidence-meta").textContent = isWeb ? `网页摘要 · ${source.url}` : source.page
    ? `PDF 第 ${source.page} 页`
    : source.chunk_id || "";
  $("evidence-text").textContent =
    source.text || "当前记录没有原文快照，可重新提问获取证据。";
  const link = isWeb ? source.url : source.file_url;
  $("pdf-link").classList.toggle("hidden", !link);
  if (link) {
    $("pdf-link").href = link;
    $("pdf-link").textContent = isWeb ? "打开来源网页 ↗" : source.page
      ? `查看 PDF 第 ${source.page} 页 ↗`
      : "查看 PDF ↗";
  }
  const scores = $("evidence-scores");
  scores.replaceChildren();
  for (const [key, label] of [
    ["distance", "向量距离"],
    ["bm25_score", "BM25"],
    ["rrf_score", "RRF"],
    ["cross_encoder_score", "Cross-Encoder"],
    ["rerank_relevance", "重排相关性"],
  ])
    if (source[key] !== undefined && source[key] !== null)
      scores.append(
        node("dt", "", label),
        node("dd", "", Number(source[key]).toFixed(5)),
      );
  $("evidence").classList.remove("hidden");
}
function openLibrary() {
  $("library-dialog").showModal();
  renderDocuments();
  refresh().catch((error) => toast(error.message));
}
function formatBytes(value) {
  return value < 1024 * 1024
    ? `${(value / 1024).toFixed(0)} KiB`
    : `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}
function renderDocuments() {
  const root = $("document-list");
  root.replaceChildren();
  if (!state.documents.length) {
    root.append(node("div", "empty-list", "上传第一篇论文，建立研读知识库。"));
    return;
  }
  for (const document of state.documents) {
    const row = node("div", "document-row");
    const info = node("div", "document-info");
    info.append(node("div", "document-name", document.name));
    const index =
      document.indexes.find(
        (item) => item.collection_name === state.collection,
      ) || document.indexes[0];
    const status =
      document.lifecycle === "deleting"
        ? "删除中，可重试"
        : stages[index?.stage || index?.status] || "等待入库";
    const meta = node(
      "div",
      "document-meta",
      `${document.page_count} 页 · ${formatBytes(document.size_bytes)}${index?.chunk_count ? ` · ${index.chunk_count} 个片段` : ""} · `,
    );
    const badge = node(
      "span",
      "document-state" + (index?.status === "failed" ? " failed" : ""),
      status,
    );
    meta.append(badge);
    info.append(meta);
    if (index?.error?.message) {
      const error = node("p", "muted", index.error.message);
      error.style.margin = "6px 0 0";
      info.append(error);
    }
    const actions = node("div", "document-actions");
    if (document.lifecycle === "active") {
      const view = node("a", "", "查看 PDF ↗");
      view.href = `/documents/${document.id}/file`;
      view.target = "_blank";
      view.rel = "noopener";
      actions.append(view);
      actions.append(
        button("重建索引", "", async () => {
          if (
            !(await confirmAction(
              "重新入库",
              `将按所选解析方式重建“${document.name}”的索引。`,
            ))
          )
            return;
          try {
            await api(
              `/documents/${document.id}/reindex`,
              jsonOptions("POST", {
                scheme: Number($("upload-scheme").value),
                refresh: true,
              }),
            );
            toast("重建任务已提交");
            await refresh();
          } catch (error) {
            toast(error.message);
          }
        }),
      );
    }
    actions.append(
      button(
        document.lifecycle === "deleting" ? "重试删除" : "删除",
        "danger",
        async () => {
          if (
            !(await confirmAction(
              "删除文档",
              `删除“${document.name}”的 PDF、向量索引和解析缓存。已有会话保留引用快照。`,
            ))
          )
            return;
          try {
            await api(`/documents/${document.id}`, { method: "DELETE" });
            toast("文档已删除");
            await refresh();
          } catch (error) {
            toast(error.message);
          }
        },
      ),
    );
    row.append(info, actions);
    root.append(row);
  }
}
function confirmAction(title, message) {
  $("confirm-title").textContent = title;
  $("confirm-message").textContent = message;
  const dialog = $("confirm-dialog");
  dialog.returnValue = "cancel";
  dialog.showModal();
  return new Promise((resolve) =>
    dialog.addEventListener(
      "close",
      () => resolve(dialog.returnValue === "confirm"),
      { once: true },
    ),
  );
}
async function refresh() {
  if (refreshRunning) return;
  refreshRunning = true;
  try {
    const [sessions, collections, documents, health] = await Promise.all([
      api("/sessions"),
      api("/collections"),
      api("/documents"),
      api("/health"),
    ]);
    state.sessions = sessions.items;
    state.collections = collections.items;
    state.documents = documents.items;
    const select = $("collection");
    const old = state.collection;
    select.replaceChildren();
    for (const collection of state.collections) {
      const option = node(
        "option",
        "",
        `${labels[collection.name] || collection.name} · ${collection.ready_chunk_count ?? collection.chunk_count}`,
      );
      option.value = collection.name;
      select.append(option);
    }
    if (!state.collections.length) {
      const option = node("option", "", "上传后选择知识库");
      option.value = "";
      select.append(option);
    }
    if (state.collections.some((item) => item.name === old)) select.value = old;
    else if (state.collections.length) state.collection = select.value;
    else state.collection = "";
    $("document-count").textContent = String(state.documents.length);
    $("status-dot").classList.add("online");
    const active = health.tasks.running + health.tasks.queued;
    $("service-status").textContent = active
      ? `${active} 个入库任务处理中`
      : "本机服务已连接";
    if (!$("pdf-files").files.length)
      $("file-hint").textContent =
        `支持批量上传 · 单文件 ${formatBytes(health.max_file_bytes)}`;
    if ($("library-dialog").open) {
      const activeIndexes = state.documents
        .flatMap((doc) => doc.indexes)
        .filter((item) => ["queued", "running"].includes(item.status));
      await Promise.all(
        activeIndexes.map(async (index) => {
          try {
            const task = await api(`/tasks/${index.task_id}`);
            index.stage = task.stage;
          } catch {}
        }),
      );
      renderDocuments();
    }
    renderSessions();
    updateComposer();
    if (!state.turns.length) renderMessages();
    if (
      state.sessionId &&
      !state.busy.has(state.sessionId) &&
      state.turns.some((turn) => turn.status === "running")
    )
      await loadSession(state.sessionId);
  } catch (error) {
    $("status-dot").classList.remove("online");
    $("service-status").textContent = "服务连接中断";
    throw error;
  } finally {
    refreshRunning = false;
  }
}

$("new-chat").addEventListener("click", newChat);
$("open-library").addEventListener("click", openLibrary);
$("refresh-documents").addEventListener("click", () =>
  refresh().catch((error) => toast(error.message)),
);
$("collection").addEventListener("change", (event) => {
  state.collection = event.target.value;
  state.documentIds = [];
  newChat();
});
$("close-evidence").addEventListener("click", () =>
  $("evidence").classList.add("hidden"),
);
$("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion($("question").value);
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendQuestion(event.target.value);
  }
});
$("question").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = Math.min(event.target.scrollHeight, 170) + "px";
});
for (const control of document.querySelectorAll("[data-close]"))
  control.addEventListener("click", () => $(control.dataset.close).close());
for (const dialog of document.querySelectorAll("dialog"))
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const box = dialog.getBoundingClientRect();
      if (
        event.clientX < box.left ||
        event.clientX > box.right ||
        event.clientY < box.top ||
        event.clientY > box.bottom
      )
        dialog.close();
    }
  });
$("settings-button").addEventListener("click", () => {
  const form = $("settings-form");
  for (const key of ["mode", "dense_k", "keyword_k", "top_k"])
    form.elements[key].value = state.options[key];
  form.elements.rewrite_query.checked = state.options.rewrite_query;
  $("settings-dialog").showModal();
});
$("settings-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.target;
  const options = {
    mode: form.elements.mode.value,
    dense_k: Number(form.elements.dense_k.value),
    keyword_k: Number(form.elements.keyword_k.value),
    top_k: Number(form.elements.top_k.value),
    rewrite_query: form.elements.rewrite_query.checked,
  };
  const available =
    options.mode === "dense"
      ? options.dense_k
      : options.mode === "bm25"
        ? options.keyword_k
        : options.dense_k + options.keyword_k;
  if (options.top_k > available) {
    toast("最终证据数需要小于或等于候选数量");
    return;
  }
  state.options = options;
  $("settings-dialog").close();
  newChat();
  toast("检索配置已应用到新会话");
});
$("scope-button").addEventListener("click", () => {
  const root = $("scope-list");
  root.replaceChildren();
  const ready = state.documents.filter(
    (doc) =>
      doc.lifecycle === "active" &&
      doc.indexes.some(
        (index) =>
          index.collection_name === state.collection &&
          index.status === "completed",
      ),
  );
  for (const doc of ready) {
    const label = node("label", "scope-option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = doc.id;
    input.checked = state.documentIds.includes(doc.id);
    const text = node("span", "", doc.name);
    text.append(node("small", "", `${doc.page_count} 页`));
    label.append(input, text);
    root.append(label);
  }
  if (!ready.length)
    root.append(
      node(
        "div",
        "empty-list",
        "该知识库尚无已登记、已完成的论文，可使用全部可用内容。",
      ),
    );
  $("scope-dialog").showModal();
});
$("scope-form").addEventListener("submit", (event) => {
  event.preventDefault();
  state.documentIds = [
    ...$("scope-list").querySelectorAll("input:checked"),
  ].map((input) => input.value);
  $("scope-dialog").close();
  newChat();
});
$("rename-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api(
      `/sessions/${renameId}`,
      jsonOptions("PATCH", { title: $("session-title").value.trim() }),
    );
    $("rename-dialog").close();
    await refresh();
  } catch (error) {
    toast(error.message);
  }
});
function updateFilesHint() {
  const files = $("pdf-files").files;
  $("file-hint").textContent = files.length
    ? `已选择 ${files.length} 份文件 · ${formatBytes([...files].reduce((sum, file) => sum + file.size, 0))}`
    : "支持批量上传 · 单文件 50 MiB";
}
$("pdf-files").addEventListener("change", updateFilesHint);
$("dropzone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("dropzone").classList.add("drag-over");
});
$("dropzone").addEventListener("dragleave", () =>
  $("dropzone").classList.remove("drag-over"),
);
$("dropzone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("dropzone").classList.remove("drag-over");
  $("pdf-files").files = event.dataTransfer.files;
  updateFilesHint();
});
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const files = $("pdf-files").files;
  if (!files.length) {
    toast("请先选择 PDF");
    return;
  }
  const form = new FormData();
  for (const file of files) form.append("files", file);
  form.append("scheme", $("upload-scheme").value);
  $("upload-button").disabled = true;
  $("upload-button").textContent = "上传中…";
  try {
    const result = await api("/documents", { method: "POST", body: form });
    $("upload-result").textContent =
      `${result.accepted_count} 份已接受，${result.rejected_count} 份被拒绝。` +
      result.items
        .filter((item) => !item.accepted)
        .map((item) => `\n${item.filename} · ${item.error.message}`)
        .join("");
    $("pdf-files").value = "";
    await refresh();
  } catch (error) {
    $("upload-result").textContent = error.message;
  } finally {
    $("upload-button").disabled = false;
    $("upload-button").textContent = "上传并入库";
  }
});
(async () => {
  try {
    await refresh();
    const last = localStorage.getItem("paperagent-session");
    if (last && state.sessions.some((item) => item.id === last))
      await loadSession(last);
    else renderMessages();
  } catch (error) {
    toast(error.message);
    renderMessages();
  }
  setInterval(() => {
    if (!document.hidden) refresh().catch(() => {});
  }, 3000);
})();

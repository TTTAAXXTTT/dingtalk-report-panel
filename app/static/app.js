(function () {
  "use strict";

  const STORAGE_KEY = "dingtalk-report-panel:config:v2";
  const API_ROOT = "/api";
  const POLL_INTERVAL_MS = 1500;
  const TERMINAL_SUCCESS = new Set(["complete", "completed", "succeeded", "success", "done"]);
  const TERMINAL_FAILURE = new Set(["failed", "failure", "error", "cancelled", "canceled"]);
  const NEEDS_ATTENTION = new Set(["needs_confirmation", "needs_attention", "uncertain", "blocked"]);
  const REPORT_DOCUMENTS = {
    completed_items: {
      title: "本周已完成事项",
      description: "仅包含有明确完成证据的事项",
      filenameSuffix: "本周已完成事项",
      tabId: "completedItemsTab",
    },
    weekly_report: {
      title: "周报",
      description: "本周工作与下周工作计划",
      filenameSuffix: "周报",
      tabId: "weeklyReportTab",
    },
  };

  const state = {
    status: null,
    conversations: [],
    selectedIds: new Set(),
    selectedMeta: new Map(),
    conversationQuery: "",
    conversationKind: "all",
    directoryWarnings: [],
    refreshWarnings: [],
    preview: null,
    previewSignature: null,
    activeJobId: null,
    activeJob: null,
    reportMarkdown: "",
    reportDocuments: {},
    activeReportDocument: "completed_items",
    reportJobId: null,
    reportBaseTitle: "钉钉聊天工作报告",
    llmConfig: null,
    llmTestPassed: false,
    llmDirty: false,
    pollTimer: null,
    saveTimer: null,
    searchTimer: null,
    restoredDataMode: null,
  };

  const el = {};

  document.addEventListener("DOMContentLoaded", initialize);

  function initialize() {
    cacheElements();
    restoreConfiguration();
    ensureDefaultTimeRange();
    bindEvents();
    updateInstructionCounter();
    updateConfigurationSummary();
    updateGenerateAvailability();

    Promise.allSettled([
      loadStatus(),
      loadConversations(),
      loadHistory(),
      loadLlmConfiguration(),
    ]).then(() => {
      renderSelectedConversations();
      updateConfigurationSummary();
    });
  }

  function cacheElements() {
    const ids = [
      "statusRefreshButton", "fixtureBanner", "connectionPill", "connectionDot", "connectionLabel",
      "conversationRefreshButton", "dwsDot", "dwsState", "dwsMessage", "dataModeValue",
      "dwsVersionValue", "conversationRefreshTime", "reportConfigForm", "conversationSearch",
      "clearConversationSearch", "selectionCount", "selectedConversationStrip", "selectedConversationChips",
      "clearConversationSelection", "conversationList", "conversationResultCount", "selectVisibleConversations",
      "conversationDirectoryWarnings", "conversationDirectoryWarningList", "manualConversationId", "addManualConversationButton",
      "startTime", "endTime", "timezone", "includeSenders", "excludeSenders", "includeKeywords",
      "excludeKeywords", "includeSystem", "includeBots", "includeRecalled", "expandReplies",
      "expandForwards", "onlyRelatedToMe", "casualFilter", "reportSubjectName", "reportTitle", "reportGrouping",
      "reportLength", "reportLanguage", "inferencePolicy", "customInstruction", "instructionCount",
      "llmProtocol", "llmBaseUrl", "llmModel", "llmApiKey", "llmApiKeyHint", "llmTimeoutSeconds", "llmMaxRetries",
      "llmTemperature", "llmMaxOutputTokens", "llmMaxInputChars", "llmMaxMessagesPerBatch", "llmAnalysisInstruction", "llmWeeklyTemplate",
      "llmSaveButton", "llmTestButton", "llmStateBadge", "llmTestMessage", "llmConnectionCheck",
      "llmModelCheck", "llmStructuredCheck", "llmSubjectCheck",
      "evidenceEnabled", "evidenceLevel", "includeFullReviewAppendix", "previewPlaceholder", "previewResult", "previewStatusIcon",
      "previewStatusTitle", "previewStatusCopy", "previewStamp", "previewConversations", "previewWindow",
      "previewFilters", "previewReport", "previewWarnings", "previewWarningsTitle", "previewWarningList", "validationSummary",
      "previewButton", "explicitConfirm", "uncertaintyAckWrap", "uncertaintyAck", "generateButton",
      "processingPanel", "processingSubtitle", "activeJobId", "progressTrack", "progressBar",
      "pipelineSteps", "countsGrid", "jobWarnings", "jobWarningList", "jobError",
      "reportPreviewPanel", "reportPreviewTitle", "reportIntegritySummary", "copyReportButton",
      "downloadReportButton", "completedItemsTab", "weeklyReportTab", "reportDocumentDescription",
      "markdownPreview", "autosaveState", "summaryConversations", "summaryStart",
      "summaryEnd", "summaryTypes", "summarySections", "historyRefreshButton", "historyList", "toastRegion",
    ];
    ids.forEach((id) => {
      el[id] = document.getElementById(id);
    });
  }

  function bindEvents() {
    el.statusRefreshButton.addEventListener("click", () => loadStatus(true));
    el.conversationRefreshButton.addEventListener("click", refreshConversations);
    el.historyRefreshButton.addEventListener("click", () => loadHistory(true));

    el.conversationSearch.addEventListener("input", () => {
      state.conversationQuery = el.conversationSearch.value.trim();
      el.clearConversationSearch.hidden = !state.conversationQuery;
      window.clearTimeout(state.searchTimer);
      state.searchTimer = window.setTimeout(loadConversations, 260);
    });

    el.clearConversationSearch.addEventListener("click", () => {
      el.conversationSearch.value = "";
      state.conversationQuery = "";
      el.clearConversationSearch.hidden = true;
      loadConversations();
      el.conversationSearch.focus();
    });

    document.querySelectorAll("[data-kind]").forEach((button) => {
      button.addEventListener("click", () => {
        state.conversationKind = button.dataset.kind || "all";
        document.querySelectorAll("[data-kind]").forEach((candidate) => {
          const active = candidate === button;
          candidate.classList.toggle("is-active", active);
          candidate.setAttribute("aria-pressed", String(active));
        });
        loadConversations();
      });
    });

    el.clearConversationSelection.addEventListener("click", () => {
      state.selectedIds.clear();
      state.selectedMeta.clear();
      renderConversations();
      configurationChanged();
    });

    el.selectVisibleConversations.addEventListener("click", () => {
      state.conversations.filter((item) => item.available).forEach((item) => {
        state.selectedIds.add(item.id);
        state.selectedMeta.set(item.id, item);
      });
      renderConversations();
      configurationChanged();
    });

    el.addManualConversationButton.addEventListener("click", addManualConversation);
    el.manualConversationId.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        addManualConversation();
      }
    });

    document.querySelectorAll("[data-range]").forEach((button) => {
      button.addEventListener("click", () => {
        applyQuickRange(button.dataset.range);
        document.querySelectorAll("[data-range]").forEach((candidate) => {
          candidate.classList.toggle("is-active", candidate === button);
        });
        configurationChanged();
      });
    });

    el.reportConfigForm.addEventListener("input", (event) => {
      if (event.target === el.explicitConfirm || event.target === el.uncertaintyAck) return;
      if (event.target === el.customInstruction) updateInstructionCounter();
      if (event.target === el.evidenceEnabled && !el.evidenceEnabled.checked) {
        el.evidenceLevel.value = "none";
      }
      if (event.target === el.evidenceLevel) {
        el.evidenceEnabled.checked = el.evidenceLevel.value !== "none";
      }
      if (event.target === el.startTime || event.target === el.endTime) {
        document.querySelectorAll("[data-range]").forEach((button) => button.classList.remove("is-active"));
      }
      configurationChanged();
    });

    el.reportConfigForm.addEventListener("change", (event) => {
      if (event.target === el.explicitConfirm || event.target === el.uncertaintyAck) return;
      if (event.target === el.timezone) {
        document.querySelectorAll("[data-range]").forEach((button) => button.classList.remove("is-active"));
      }
      configurationChanged();
    });

    el.previewButton.addEventListener("click", previewScope);
    el.explicitConfirm.addEventListener("change", updateGenerateAvailability);
    el.uncertaintyAck.addEventListener("change", updateGenerateAvailability);
    el.generateButton.addEventListener("click", createJob);
    el.copyReportButton.addEventListener("click", copyReport);
    el.downloadReportButton.addEventListener("click", downloadReport);
    el.llmSaveButton.addEventListener("click", () => saveLlmConfiguration(false));
    el.llmTestButton.addEventListener("click", testLlmConfiguration);
    [
      el.llmProtocol, el.llmBaseUrl, el.llmModel, el.llmApiKey, el.llmTimeoutSeconds, el.llmMaxRetries,
      el.llmTemperature, el.llmMaxOutputTokens, el.llmMaxInputChars, el.llmMaxMessagesPerBatch, el.llmAnalysisInstruction, el.llmWeeklyTemplate,
    ].forEach((field) => {
      field.addEventListener("input", llmConfigurationChanged);
      field.addEventListener("change", llmConfigurationChanged);
    });
    document.querySelectorAll("[data-report-document]").forEach((button) => {
      button.addEventListener("click", () => switchReportDocument(button.dataset.reportDocument));
    });
  }

  async function apiFetch(path, options) {
    const requestPath = path.startsWith(`${API_ROOT}/`)
      ? path
      : `${API_ROOT}${path.startsWith("/") ? path : `/${path}`}`;
    const response = await fetch(requestPath, {
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        ...(options && options.body ? { "Content-Type": "application/json" } : {}),
        ...((options && options.headers) || {}),
      },
      ...options,
    });

    const contentType = response.headers.get("content-type") || "";
    let payload;
    if (contentType.includes("application/json")) {
      payload = await response.json().catch(() => ({}));
    } else {
      const text = await response.text();
      payload = text ? { message: text } : {};
    }

    if (!response.ok) {
      const message = errorMessage(payload, `请求失败（HTTP ${response.status}）`);
      const error = new Error(message);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload || {};
  }

  function buildLlmConfiguration(includeApiKey) {
    const temperatureText = el.llmTemperature.value.trim();
    const config = {
      protocol: el.llmProtocol.value || "openai_responses",
      base_url: el.llmBaseUrl.value.trim(),
      model: el.llmModel.value.trim(),
      timeout_seconds: Number(el.llmTimeoutSeconds.value),
      max_retries: Number(el.llmMaxRetries.value),
      temperature: temperatureText === "" ? null : Number(temperatureText),
      max_output_tokens: Number(el.llmMaxOutputTokens.value),
      max_input_chars: Number(el.llmMaxInputChars.value),
      max_messages_per_batch: Number(el.llmMaxMessagesPerBatch.value),
      analysis_instruction: el.llmAnalysisInstruction.value.trim(),
      weekly_report_template: el.llmWeeklyTemplate.value.trim(),
    };
    if (includeApiKey && el.llmApiKey.value.trim()) config.api_key = el.llmApiKey.value.trim();
    return config;
  }

  function validateLlmConfiguration(config) {
    const errors = [];
    if (!["openai_responses", "openai_chat_completions"].includes(config.protocol)) errors.push("请选择支持的接口协议。");
    if (!config.base_url) {
      errors.push("请填写 Base URL。");
    } else {
      try {
        const url = new URL(config.base_url);
        if (!["http:", "https:"].includes(url.protocol)) errors.push("Base URL 必须使用 http 或 https。");
      } catch (_) {
        errors.push("Base URL 格式不正确。");
      }
    }
    if (!config.model) errors.push("请填写模型名称。");
    if (!Number.isFinite(config.timeout_seconds) || config.timeout_seconds < 10 || config.timeout_seconds > 600) errors.push("请求超时必须在 10–600 秒之间。");
    if (!Number.isInteger(config.max_retries) || config.max_retries < 0 || config.max_retries > 3) errors.push("瞬时失败重试次数必须是 0–3 的整数。");
    if (config.temperature !== null && (!Number.isFinite(config.temperature) || config.temperature < 0 || config.temperature > 2)) errors.push("Temperature 留空使用自动模式，手动填写时必须在 0–2 之间。");
    if (!Number.isInteger(config.max_output_tokens) || config.max_output_tokens < 512 || config.max_output_tokens > 128000) errors.push("最大输出 Tokens 必须是 512–128000 的整数。");
    if (!Number.isInteger(config.max_input_chars) || config.max_input_chars < 2000 || config.max_input_chars > 1000000) errors.push("每批最大输入字符必须是 2000–1000000 的整数。");
    if (!Number.isInteger(config.max_messages_per_batch) || config.max_messages_per_batch < 1 || config.max_messages_per_batch > 100) errors.push("每批最大消息数必须是 1–100 的整数。");
    if (!config.analysis_instruction) errors.push("请填写大模型分析要求。");
    if (!config.weekly_report_template) errors.push("请填写周报格式模板。");
    return errors;
  }

  async function loadLlmConfiguration() {
    renderLlmLoading();
    try {
      const raw = await apiFetch("/llm/config");
      const config = normalizeLlmConfiguration(raw);
      state.llmConfig = config;
      state.llmDirty = false;
      state.llmTestPassed = config.test.ok;
      applyLlmConfiguration(config);
      renderLlmTest(config.test);
      updateGenerateAvailability();
    } catch (error) {
      state.llmConfig = null;
      state.llmDirty = false;
      state.llmTestPassed = false;
      renderLlmUnavailable(error);
      updateGenerateAvailability();
    }
  }

  function normalizeLlmConfiguration(raw) {
    const source = isPlainObject(raw.config) ? raw.config : raw;
    const testSource = isPlainObject(raw.test_status)
      ? raw.test_status
      : isPlainObject(raw.last_test)
        ? raw.last_test
        : isPlainObject(source.test_status)
          ? source.test_status
          : isPlainObject(source.last_test)
            ? source.last_test
            : { ...source, ...raw };
    return {
      protocol: firstString(source.protocol, source.api_protocol, "openai_responses"),
      baseUrl: firstString(source.base_url, source.baseUrl, "https://api.openai.com/v1"),
      model: firstString(source.model, source.model_name, ""),
      timeoutSeconds: finiteNumber(source.timeout_seconds, source.timeout, 120),
      maxRetries: finiteNumber(source.max_retries, 1),
      temperature: nullableFiniteNumber(source.temperature, null),
      maxOutputTokens: finiteNumber(source.max_output_tokens, source.max_tokens, 6000),
      maxInputChars: finiteNumber(source.max_input_chars, 8000),
      maxMessagesPerBatch: finiteNumber(source.max_messages_per_batch, 3),
      analysisInstruction: firstString(source.analysis_instruction, source.analysis_prompt, el.llmAnalysisInstruction.value),
      weeklyReportTemplate: firstString(source.weekly_report_template, source.report_template, el.llmWeeklyTemplate.value),
      apiKeyConfigured: firstDefinedBoolean(
        source.api_key_configured,
        source.api_key_set,
        raw.api_key_configured,
        raw.api_key_set,
        false,
      ) === true,
      test: normalizeLlmTest(testSource),
    };
  }

  function normalizeLlmTest(raw) {
    const source = isPlainObject(raw.result) ? raw.result : (isPlainObject(raw) ? raw : {});
    const checks = isPlainObject(source.checks) ? source.checks : {};
    const connection = firstDefinedBoolean(
      source.connection_ok,
      source.connection,
      source.connectivity_ok,
      source.connection && source.connection.ok,
      checks.connection && checks.connection.ok,
      checks.connectivity && checks.connectivity.ok,
      checks.connection,
      checks.connectivity,
    );
    const model = firstDefinedBoolean(
      source.model_ok,
      source.model_response_ok,
      source.model && source.model.ok,
      source.model_response && source.model_response.ok,
      checks.model && checks.model.ok,
      checks.model_response && checks.model_response.ok,
      checks.model,
      checks.model_response,
    );
    const structured = firstDefinedBoolean(
      source.structured_output_ok,
      source.schema_ok,
      source.structured_output && source.structured_output.ok,
      source.schema && source.schema.ok,
      checks.structured_output && checks.structured_output.ok,
      checks.schema && checks.schema.ok,
      checks.structured_output,
      checks.schema,
    );
    const subjectIdentity = firstDefinedBoolean(
      source.subject_identity_validation_validated,
      source.subject_identity_ok,
      checks.subject_identity && checks.subject_identity.ok,
      checks.subject_identity,
    );
    const overall = firstDefinedBoolean(source.ok, source.success, source.passed, source.last_test_ok, raw && raw.last_test_ok);
    const inferred = overall === true;
    const normalized = {
      connectionOk: connection == null ? (inferred ? true : null) : connection,
      modelOk: model == null ? (inferred ? true : null) : model,
      structuredOk: structured == null ? (inferred ? true : null) : structured,
      subjectIdentityOk: subjectIdentity == null ? (inferred ? true : null) : subjectIdentity,
      message: firstString(source.message, source.detail, source.error, source.last_test_message, ""),
      testedAt: firstString(source.tested_at, source.last_tested_at, source.timestamp, ""),
    };
    normalized.ok = normalized.connectionOk === true && normalized.modelOk === true && normalized.structuredOk === true && normalized.subjectIdentityOk === true && overall !== false;
    return normalized;
  }

  function applyLlmConfiguration(config) {
    setValue(el.llmProtocol, config.protocol);
    el.llmBaseUrl.value = config.baseUrl;
    el.llmModel.value = config.model;
    el.llmTimeoutSeconds.value = String(config.timeoutSeconds);
    el.llmMaxRetries.value = String(config.maxRetries);
    el.llmTemperature.value = config.temperature == null ? "" : String(config.temperature);
    el.llmMaxOutputTokens.value = String(config.maxOutputTokens);
    el.llmMaxInputChars.value = String(config.maxInputChars);
    el.llmMaxMessagesPerBatch.value = String(config.maxMessagesPerBatch);
    el.llmAnalysisInstruction.value = config.analysisInstruction;
    el.llmWeeklyTemplate.value = config.weeklyReportTemplate;
    el.llmApiKey.value = "";
    el.llmApiKey.placeholder = config.apiKeyConfigured
      ? "已安全保存；留空表示保留现有 Key"
      : "如服务商需要鉴权则填写；本地免鉴权服务可留空";
    el.llmApiKeyHint.textContent = config.apiKeyConfigured
      ? "已保存 API Key；页面不会回显密钥。要更换时输入新 Key 后保存。"
      : "尚未配置 API Key；如你的模型服务需要鉴权，请输入后保存。本地免鉴权服务可留空。";
  }

  function llmConfigurationChanged() {
    state.llmDirty = true;
    state.llmTestPassed = false;
    renderLlmTest({
      connectionOk: null,
      modelOk: null,
      structuredOk: null,
      subjectIdentityOk: null,
      ok: false,
      message: "配置已修改。请保存并重新测试，测试通过前不能生成。",
      testedAt: "",
    });
    invalidatePreview();
  }

  async function saveLlmConfiguration(silent) {
    const config = buildLlmConfiguration(true);
    const errors = validateLlmConfiguration(config);
    if (errors.length) {
      showToast("模型配置不完整", errors.join("；"), "error", 9000);
      return false;
    }

    setButtonLoading(el.llmSaveButton, true, "正在保存");
    try {
      const raw = await apiFetch("/llm/config", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      const rawConfig = isPlainObject(raw.config) ? raw.config : raw;
      const merged = {
        ...config,
        ...rawConfig,
        api_key_configured: firstDefinedBoolean(
          rawConfig.api_key_configured,
          rawConfig.api_key_set,
          raw.api_key_configured,
          raw.api_key_set,
          Boolean(config.api_key) || Boolean(state.llmConfig && state.llmConfig.apiKeyConfigured),
        ),
      };
      delete merged.api_key;
      const normalized = normalizeLlmConfiguration({ ...raw, config: merged });
      state.llmConfig = normalized;
      state.llmDirty = false;
      state.llmTestPassed = normalized.test.ok;
      applyLlmConfiguration(normalized);
      renderLlmTest(normalized.test.ok ? normalized.test : {
        connectionOk: null,
        modelOk: null,
        structuredOk: null,
        subjectIdentityOk: null,
        ok: false,
        message: "配置已保存。请测试接口、模型响应、结构化输出和本人识别。",
        testedAt: "",
      });
      invalidatePreview();
      if (!silent) showToast("模型配置已保存", "保存不会读取聊天或生成报告；请继续测试模型。", "success");
      return true;
    } catch (error) {
      showToast("模型配置保存失败", error.message, "error", 9000);
      return false;
    } finally {
      setButtonLoading(el.llmSaveButton, false);
      updateGenerateAvailability();
    }
  }

  async function testLlmConfiguration() {
    setButtonLoading(el.llmTestButton, true, "正在测试");
    try {
      if (state.llmDirty || !state.llmConfig) {
        const saved = await saveLlmConfiguration(true);
        if (!saved) return;
      }
      state.llmTestPassed = false;
      renderLlmTesting();
      const raw = await apiFetch("/llm/test", { method: "POST" });
      const result = normalizeLlmTest(raw);
      state.llmTestPassed = result.ok;
      if (state.llmConfig) state.llmConfig.test = result;
      renderLlmTest(result);
      invalidatePreview();
      if (result.ok) {
        showToast("模型测试通过", "接口连接、模型响应、结构化输出和本人识别均已通过。请重新预览范围后生成。", "success", 7500);
      } else {
        showToast("模型测试未通过", result.message || "请查看四项测试状态并检查配置。", "error", 10000);
      }
    } catch (error) {
      state.llmTestPassed = false;
      renderLlmTest({
        connectionOk: false,
        modelOk: false,
        structuredOk: false,
        subjectIdentityOk: false,
        ok: false,
        message: error.message,
        testedAt: "",
      });
      invalidatePreview();
      showToast("模型测试失败", error.message, "error", 10000);
    } finally {
      setButtonLoading(el.llmTestButton, false);
      updateGenerateAvailability();
    }
  }

  function renderLlmLoading() {
    el.llmStateBadge.className = "model-state-badge is-pending";
    el.llmStateBadge.textContent = "正在读取";
    el.llmTestMessage.textContent = "正在读取本机服务端保存的模型配置…";
    setModelCheck(el.llmConnectionCheck, null);
    setModelCheck(el.llmModelCheck, null);
    setModelCheck(el.llmStructuredCheck, null);
    setModelCheck(el.llmSubjectCheck, null);
  }

  function renderLlmTesting() {
    el.llmStateBadge.className = "model-state-badge is-testing";
    el.llmStateBadge.textContent = "测试中";
    el.llmTestMessage.textContent = "正在使用无真实聊天内容的测试样例核对四项能力…";
    setModelCheck(el.llmConnectionCheck, null, "测试中");
    setModelCheck(el.llmModelCheck, null, "等待响应");
    setModelCheck(el.llmStructuredCheck, null, "等待校验");
    setModelCheck(el.llmSubjectCheck, null, "等待校验");
  }

  function renderLlmTest(result) {
    const safe = result || {};
    setModelCheck(el.llmConnectionCheck, safe.connectionOk);
    setModelCheck(el.llmModelCheck, safe.modelOk);
    setModelCheck(el.llmStructuredCheck, safe.structuredOk);
    setModelCheck(el.llmSubjectCheck, safe.subjectIdentityOk);
    const failed = safe.connectionOk === false || safe.modelOk === false || safe.structuredOk === false || safe.subjectIdentityOk === false;
    el.llmStateBadge.className = `model-state-badge ${safe.ok ? "is-passed" : failed ? "is-failed" : "is-pending"}`;
    el.llmStateBadge.textContent = safe.ok ? "测试通过" : failed ? "测试未通过" : "尚未通过";
    const timeCopy = safe.testedAt ? `（${formatDateTime(safe.testedAt)}）` : "";
    el.llmTestMessage.textContent = safe.message || (safe.ok
      ? `四项测试均已通过${timeCopy}。配置变化后需要重新测试。`
      : "请保存配置，再测试模型。测试通过前不能正式生成。");
  }

  function renderLlmUnavailable(error) {
    el.llmStateBadge.className = "model-state-badge is-failed";
    el.llmStateBadge.textContent = "配置不可用";
    el.llmTestMessage.textContent = `无法读取模型配置：${error.message}`;
    setModelCheck(el.llmConnectionCheck, false);
    setModelCheck(el.llmModelCheck, false);
    setModelCheck(el.llmStructuredCheck, false);
    setModelCheck(el.llmSubjectCheck, false);
  }

  function setModelCheck(item, status, pendingCopy) {
    item.dataset.status = status === true ? "passed" : status === false ? "failed" : "pending";
    const icon = item.querySelector("span");
    icon.textContent = status === true ? "✓" : status === false ? "×" : "—";
    if (pendingCopy) item.title = pendingCopy;
    else item.removeAttribute("title");
  }

  async function loadStatus(showFeedback) {
    setStatusLoading();
    try {
      const raw = await apiFetch("/status");
      const status = normalizeStatus(raw);
      state.status = status;
      renderStatus(status);
      reconcileRestoredDataMode(status.dataMode);
      if (showFeedback) showToast("连接状态已更新", status.authenticated ? "当前钉钉已连接。" : status.message || "当前尚未连接。", status.authenticated ? "success" : "warning");
    } catch (error) {
      state.status = null;
      renderStatusError(error);
      if (showFeedback) showToast("无法检查连接", error.message, "error");
    }
  }

  function normalizeStatus(raw) {
    const dws = isPlainObject(raw.dws) ? raw.dws : raw;
    const dataMode = firstString(raw.data_mode, raw.mode, "unknown");
    return {
      appVersion: firstString(raw.app_version, raw.version, "—"),
      dataMode,
      generationMode: firstString(raw.generation_mode, "manual"),
      scheduled: raw.scheduled === true,
      installed: dws.installed === true,
      version: firstString(dws.version, dws.dws_version, "—"),
      authenticated: dws.authenticated === true,
      authCommandOk: dws.auth_command_ok === true,
      message: firstString(dws.message, raw.message, ""),
    };
  }

  function setStatusLoading() {
    el.connectionDot.className = "status-dot status-dot-pending";
    el.dwsDot.className = "status-dot status-dot-pending";
    el.connectionLabel.textContent = "正在检查…";
    el.dwsState.textContent = "正在读取连接状态";
  }

  function renderStatus(status) {
    const connected = status.installed && status.authenticated && status.authCommandOk;
    el.connectionDot.className = `status-dot ${connected ? "status-dot-ok" : "status-dot-error"}`;
    el.dwsDot.className = `status-dot ${connected ? "status-dot-ok" : "status-dot-error"}`;
    el.connectionLabel.textContent = connected ? "已连接" : "需要连接";
    el.dwsState.textContent = connected ? "当前钉钉连接可用" : status.installed ? "钉钉尚未完成连接" : "钉钉数据工具不可用";
    el.dwsMessage.textContent = status.message || (connected ? "可以刷新正常聊天列表。" : "请完成钉钉连接后重新检查。此页面不会保存账号密码。" );
    el.dataModeValue.textContent = dataModeLabel(status.dataMode);
    el.dwsVersionValue.textContent = status.version;

    const fixture = /fixture|test|demo|mock/i.test(status.dataMode);
    el.fixtureBanner.hidden = !fixture;
    document.body.classList.toggle("fixture-mode", fixture);

    if (status.scheduled || status.generationMode.toLowerCase() !== "manual") {
      showToast("运行模式异常", "服务端报告了非手动运行配置。页面不会主动触发生成，请检查后端设置。", "error", 9000);
    }
  }

  function renderStatusError(error) {
    el.connectionDot.className = "status-dot status-dot-error";
    el.dwsDot.className = "status-dot status-dot-error";
    el.connectionLabel.textContent = "状态不可用";
    el.dwsState.textContent = "无法连接本地服务";
    el.dwsMessage.textContent = error.message || "请确认本地服务已启动。";
    el.dataModeValue.textContent = "未知";
    el.dwsVersionValue.textContent = "—";
  }

  function dataModeLabel(mode) {
    if (/fixture|test|demo|mock/i.test(mode)) return "测试数据";
    if (/real|live|production/i.test(mode)) return "真实数据";
    return mode && mode !== "unknown" ? mode : "未知";
  }

  async function refreshConversations() {
    setButtonLoading(el.conversationRefreshButton, true, "正在刷新");
    try {
      const result = await apiFetch("/conversations/refresh", {
        method: "POST",
        body: JSON.stringify({}),
      });
      el.conversationRefreshTime.textContent = formatDateTime(firstString(result.refreshed_at, result.updated_at, new Date().toISOString()));
      await loadConversations();
      const warnings = normalizeWarnings(result.warnings);
      state.refreshWarnings = warnings;
      state.directoryWarnings = uniqueStrings([...state.refreshWarnings, ...state.directoryWarnings]);
      renderDirectoryWarnings();
      showToast("聊天列表已刷新", warnings.length ? `已刷新，但有 ${warnings.length} 项提示需要查看。` : "已读取当前钉钉的正常聊天。", warnings.length ? "warning" : "success");
    } catch (error) {
      showToast("刷新聊天失败", error.message, "error", 8500);
    } finally {
      setButtonLoading(el.conversationRefreshButton, false);
    }
  }

  async function loadConversations() {
    el.conversationList.setAttribute("aria-busy", "true");
    el.conversationList.replaceChildren(makeLoadingState("正在加载聊天列表", "这里只会显示当前钉钉的正常对话。"));
    const params = new URLSearchParams();
    if (state.conversationQuery) params.set("query", state.conversationQuery);
    if (state.conversationKind !== "all") params.set("kind", state.conversationKind);
    const suffix = params.toString() ? `?${params}` : "";
    try {
      const raw = await apiFetch(`/conversations${suffix}`);
      const items = arrayFrom(raw, ["items", "conversations", "results", "data"]);
      state.conversations = items.map(normalizeConversation).filter((item) => item.id);
      state.conversations.forEach((item) => {
        if (state.selectedIds.has(item.id)) state.selectedMeta.set(item.id, item);
      });
      renderConversations();
      const warnings = normalizeWarnings(raw.warnings);
      state.directoryWarnings = uniqueStrings([...state.refreshWarnings, ...warnings]);
      renderDirectoryWarnings();
      if (warnings.length) showToast("会话目录有提示", warnings[0], "warning", 7000);
    } catch (error) {
      state.conversations = [];
      renderConversationError(error);
    } finally {
      el.conversationList.setAttribute("aria-busy", "false");
    }
  }

  function normalizeConversation(item, index) {
    const rawKind = firstString(item.kind, item.type, item.conversation_type, "unknown").toLowerCase();
    let kind = rawKind;
    if (["1", "direct", "single", "private", "dm", "one_to_one"].includes(rawKind)) kind = "direct";
    if (["2", "group", "chat", "multi", "group_chat"].includes(rawKind)) kind = "group";
    const participants = Array.isArray(item.participants)
      ? item.participants.map((person) => typeof person === "string" ? person : firstString(person.name, person.display_name, person.title)).filter(Boolean)
      : splitList(firstString(item.participants, item.members, ""));
    return {
      id: firstString(item.id, item.conversation_id, item.open_conversation_id, item.cid, index ? "" : ""),
      title: firstString(item.title, item.name, item.display_name, item.conversation_name, "未命名会话"),
      kind,
      lastActive: firstString(item.last_active, item.last_message_at, item.updated_at, ""),
      participants,
      available: item.available !== false && item.accessible !== false && !["unavailable", "forbidden", "denied"].includes(firstString(item.status, "").toLowerCase()),
      source: firstString(item.source, ""),
      warning: firstString(item.warning, item.uncertainty, item.error, ""),
    };
  }

  function renderConversations() {
    el.conversationList.replaceChildren();
    if (!state.conversations.length) {
      const title = state.conversationQuery ? "没有找到匹配的聊天" : "当前没有可显示的聊天";
      const copy = state.conversationQuery ? "请调整搜索词或会话类型。" : "请检查连接后刷新聊天列表。";
      el.conversationList.append(makeEmptyState(title, copy));
    } else {
      const fragment = document.createDocumentFragment();
      state.conversations.forEach((item) => fragment.append(makeConversationRow(item)));
      el.conversationList.append(fragment);
    }
    el.conversationResultCount.textContent = `显示 ${state.conversations.length} 个会话，可用 ${state.conversations.filter((item) => item.available).length} 个`;
    renderSelectedConversations();
  }

  function makeConversationRow(item) {
    const label = document.createElement("label");
    label.className = "conversation-card";
    if (state.selectedIds.has(item.id)) label.classList.add("is-selected");
    if (!item.available) label.classList.add("is-unavailable");
    label.title = item.warning || (!item.available ? "当前会话不可读取" : item.title);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selectedIds.has(item.id);
    checkbox.disabled = !item.available;
    checkbox.setAttribute("aria-label", `选择会话：${item.title}`);
    checkbox.addEventListener("change", () => toggleConversation(item, checkbox));

    const avatar = document.createElement("span");
    avatar.className = `conversation-avatar ${item.kind === "group" ? "is-group" : ""}`;
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = conversationInitial(item.title, item.kind);

    const copy = document.createElement("span");
    copy.className = "conversation-copy";
    const title = document.createElement("strong");
    title.textContent = item.title;
    const description = document.createElement("small");
    const participants = item.participants.slice(0, 4).join("、");
    description.textContent = participants || (item.kind === "group" ? "群聊" : item.kind === "direct" ? "单聊" : "普通会话");
    copy.append(title, description);

    const meta = document.createElement("span");
    meta.className = "conversation-meta";
    const time = document.createElement("span");
    time.textContent = item.lastActive ? formatDateTime(item.lastActive, true) : "暂无活动时间";
    const kind = document.createElement("span");
    kind.className = item.available ? "conversation-kind" : "conversation-unavailable";
    kind.textContent = item.available ? kindLabel(item.kind) : "不可读取";
    meta.append(time, kind);

    label.append(checkbox, avatar, copy, meta);
    return label;
  }

  function toggleConversation(item, checkbox) {
    if (!item.available) return;
    const selected = checkbox.checked;
    if (selected) {
      state.selectedIds.add(item.id);
      state.selectedMeta.set(item.id, item);
    } else {
      state.selectedIds.delete(item.id);
      state.selectedMeta.delete(item.id);
    }
    checkbox.closest(".conversation-card")?.classList.toggle("is-selected", selected);
    renderSelectedConversations();
    configurationChanged();
  }

  function addManualConversation() {
    const id = el.manualConversationId.value.trim();
    if (!id) {
      showToast("请填写会话 ID", "支持粘贴钉钉 openConversationId。", "warning");
      el.manualConversationId.focus();
      return;
    }
    if (id.length > 512 || /[\s\u0000-\u001f]/.test(id)) {
      showToast("会话 ID 格式无效", "会话 ID 不能包含空格或控制字符，且长度不能超过 512。", "error");
      el.manualConversationId.focus();
      return;
    }

    const known = state.conversations.find((item) => item.id === id) || state.selectedMeta.get(id);
    const item = known || {
      id,
      title: `手动会话 · ${abbreviateId(id)}`,
      kind: "unknown",
      lastActive: "",
      participants: [],
      available: true,
      source: "manual",
      warning: "该会话尚未由目录验证，将在预览和采集阶段严格核对。",
    };
    state.selectedIds.add(id);
    state.selectedMeta.set(id, item);
    el.manualConversationId.value = "";
    renderConversations();
    configurationChanged();
    showToast(known ? "会话已选择" : "已添加待验证会话", known ? item.title : "预览和采集时将核对 ID 与访问权限。", known ? "success" : "warning");
  }

  function renderDirectoryWarnings() {
    el.conversationDirectoryWarningList.replaceChildren();
    el.conversationDirectoryWarnings.hidden = !state.directoryWarnings.length;
    if (!state.directoryWarnings.length) return;
    const fragment = document.createDocumentFragment();
    state.directoryWarnings.forEach((warning) => {
      const item = document.createElement("li");
      item.textContent = warning;
      fragment.append(item);
    });
    el.conversationDirectoryWarningList.append(fragment);
  }

  function renderSelectedConversations() {
    const items = Array.from(state.selectedIds).map((id) => state.selectedMeta.get(id) || { id, title: id, kind: "unknown" });
    el.selectionCount.textContent = `已选 ${items.length} 个`;
    el.selectedConversationStrip.hidden = !items.length;
    el.selectedConversationChips.replaceChildren();

    const fragment = document.createDocumentFragment();
    items.forEach((item) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const text = document.createElement("span");
      text.textContent = item.title;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.setAttribute("aria-label", `移除${item.title}`);
      remove.textContent = "×";
      remove.addEventListener("click", () => {
        state.selectedIds.delete(item.id);
        state.selectedMeta.delete(item.id);
        renderConversations();
        configurationChanged();
      });
      chip.append(text, remove);
      fragment.append(chip);
    });
    el.selectedConversationChips.append(fragment);
  }

  function renderConversationError(error) {
    el.conversationList.replaceChildren(makeEmptyState("无法读取聊天列表", error.message || "请检查当前钉钉连接。"));
    el.conversationResultCount.textContent = "会话加载失败";
    renderSelectedConversations();
  }

  function conversationInitial(title, kind) {
    if (kind === "group") return "群";
    const clean = String(title || "聊").trim();
    return Array.from(clean)[0] || "聊";
  }

  function kindLabel(kind) {
    if (kind === "direct") return "单聊";
    if (kind === "group") return "群聊";
    return "会话";
  }

  function ensureDefaultTimeRange() {
    if (el.startTime.value && el.endTime.value) return;
    applyQuickRange("lastWeek");
    const defaultButton = document.querySelector('[data-range="lastWeek"]');
    if (defaultButton) defaultButton.classList.add("is-active");
    saveConfigurationSoon();
  }

  function applyQuickRange(range) {
    const now = zonedPseudoDate(new Date(), el.timezone.value || "Asia/Shanghai");
    let start = new Date(now.getTime());
    let end = new Date(now.getTime());

    if (range === "today") {
      start.setUTCHours(0, 0, 0, 0);
    } else if (range === "last7") {
      start.setUTCDate(start.getUTCDate() - 7);
    } else if (range === "thisWeek") {
      start = startOfWeek(start);
    } else {
      end = startOfWeek(end);
      start = new Date(end.getTime());
      start.setUTCDate(start.getUTCDate() - 7);
    }

    el.startTime.value = pseudoDateTimeLocal(start);
    el.endTime.value = pseudoDateTimeLocal(end);
  }

  function startOfWeek(date) {
    const result = new Date(date.getTime());
    const daysFromMonday = (result.getUTCDay() + 6) % 7;
    result.setUTCDate(result.getUTCDate() - daysFromMonday);
    result.setUTCHours(0, 0, 0, 0);
    return result;
  }

  function zonedPseudoDate(date, timeZone) {
    try {
      const parts = new Intl.DateTimeFormat("en-CA", {
        timeZone,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hourCycle: "h23",
      }).formatToParts(date).reduce((result, part) => {
        if (part.type !== "literal") result[part.type] = Number(part.value);
        return result;
      }, {});
      return new Date(Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second));
    } catch (_) {
      return new Date(date.getTime());
    }
  }

  function pseudoDateTimeLocal(date) {
    const pad = (value) => String(value).padStart(2, "0");
    return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}T${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`;
  }

  function configurationChanged() {
    invalidatePreview();
    updateInstructionCounter();
    updateConfigurationSummary();
    saveConfigurationSoon();
  }

  function invalidatePreview() {
    state.preview = null;
    state.previewSignature = null;
    el.previewPlaceholder.hidden = false;
    el.previewResult.hidden = true;
    el.validationSummary.hidden = true;
    el.explicitConfirm.checked = false;
    el.explicitConfirm.disabled = true;
    el.uncertaintyAck.checked = false;
    el.uncertaintyAckWrap.hidden = true;
    updateGenerateAvailability();
  }

  function buildConfiguration() {
    const messageTypes = checkedValues('input[name="messageType"]');
    const sections = checkedValues('input[name="reportSection"]');
    const evidenceEnabled = true;
    return {
      conversation_ids: Array.from(state.selectedIds),
      start: normalizeLocalDateTime(el.startTime.value),
      end: normalizeLocalDateTime(el.endTime.value),
      timezone: el.timezone.value || "Asia/Shanghai",
      filters: {
        message_types: messageTypes,
        include_senders: splitList(el.includeSenders.value),
        exclude_senders: splitList(el.excludeSenders.value),
        include_keywords: splitList(el.includeKeywords.value),
        exclude_keywords: splitList(el.excludeKeywords.value),
        include_system_messages: el.includeSystem.checked,
        include_bot_messages: el.includeBots.checked,
        include_recalled: el.includeRecalled.checked,
        expand_replies: false,
        expand_forwarded: false,
        only_related_to_me: el.onlyRelatedToMe.checked,
        casual_filter: el.casualFilter.value === "aggressive" ? "conservative" : el.casualFilter.value,
      },
      report: {
        subject_name: el.reportSubjectName.value.trim(),
        title: el.reportTitle.value.trim() || "钉钉聊天工作报告",
        grouping: el.reportGrouping.value,
        sections,
        length: el.reportLength.value,
        max_length_chars: reportLengthCharacters(el.reportLength.value),
        language: "zh-CN",
        inference_policy: "facts_only",
        custom_instruction: el.customInstruction.value.trim(),
        evidence_enabled: evidenceEnabled,
        evidence_level: "all_items",
        include_full_review_appendix: false,
      },
    };
  }

  function validateConfiguration(config) {
    const errors = [];
    if (!config.conversation_ids.length) errors.push("请至少选择一个可读取的聊天会话。");
    if (!config.start) errors.push("请填写开始时间。");
    if (!config.end) errors.push("请填写结束时间。");
    if (config.start && config.end && config.start >= config.end) errors.push("结束时间必须晚于开始时间。");
    if (!config.filters.message_types.length) errors.push("请至少选择一种消息类型。");
    if (!config.report.subject_name) errors.push("请填写报告本人姓名，并与钉钉发送人显示名保持一致。");
    if (["我", "本人", "自己", "me", "myself"].includes(config.report.subject_name.toLowerCase())) errors.push("报告本人不能只写“我”或“本人”，请填写具体姓名。");
    if (!config.report.sections.length) errors.push("请至少选择一个报告章节。");
    if (!config.report.custom_instruction) errors.push("请填写本次报告的具体要求。");
    return errors;
  }

  async function previewScope() {
    const config = buildConfiguration();
    const errors = validateConfiguration(config);
    if (errors.length) {
      renderValidationErrors(errors);
      focusFirstInvalid(config);
      return;
    }

    el.validationSummary.hidden = true;
    setButtonLoading(el.previewButton, true, "正在核对范围");
    try {
      const raw = await apiFetch("/preview", {
        method: "POST",
        body: JSON.stringify(config),
      });
      const preview = normalizePreview(raw, config);
      state.preview = preview;
      state.previewSignature = stableStringify(config);
      renderPreview(preview, config);
      el.explicitConfirm.disabled = !preview.canGenerate;
      el.explicitConfirm.checked = false;
      el.uncertaintyAck.checked = false;
      el.uncertaintyAckWrap.hidden = !preview.canGenerate || !preview.warnings.length;
      updateGenerateAvailability();
      showToast(preview.canGenerate ? "范围预览完成" : "当前范围不能生成", preview.canGenerate ? "请核对范围并进行明确确认。" : "请先处理页面中列出的问题。", preview.canGenerate ? (preview.warnings.length ? "warning" : "success") : "error");
    } catch (error) {
      invalidatePreview();
      renderValidationErrors([error.message || "预览范围失败。"]);
      showToast("预览失败", error.message, "error", 8500);
    } finally {
      setButtonLoading(el.previewButton, false);
    }
  }

  function normalizePreview(raw, config) {
    const resolvedRaw = arrayFrom(raw, ["resolved_conversations", "conversations", "resolved"]);
    const resolved = resolvedRaw.length
      ? resolvedRaw.map((item, index) => typeof item === "string" ? { id: item, title: state.selectedMeta.get(item)?.title || item, available: true } : normalizeConversation(item, index))
      : config.conversation_ids.map((id) => state.selectedMeta.get(id) || { id, title: id, available: true, kind: "unknown", participants: [] });
    const warnings = normalizeWarnings(raw.warnings || raw.uncertainties || raw.warning);
    warnings.push(...normalizeWarnings(raw.blocking_errors));
    const failed = arrayFrom(raw, ["failed_conversations", "unavailable_conversations"]);
    failed.forEach((item) => {
      const title = typeof item === "string" ? item : firstString(item.title, item.name, item.id, item.conversation_id, "未知会话");
      const reason = typeof item === "string" ? "未能解析或访问" : firstString(item.reason, item.error, item.message, "未能解析或访问");
      warnings.push(`${title}：${reason}`);
    });
    const resolvedIds = new Set(resolved.map((item) => item.id).filter(Boolean));
    const missingIds = config.conversation_ids.filter((id) => !resolvedIds.has(id));
    missingIds.forEach((id) => warnings.push(`${state.selectedMeta.get(id)?.title || id}：预览结果未确认该会话。`));

    let canGenerate;
    if (typeof raw.can_generate === "boolean") {
      canGenerate = raw.can_generate;
    } else {
      canGenerate = resolved.length === config.conversation_ids.length && missingIds.length === 0 && failed.length === 0;
    }
    if (!state.llmTestPassed) {
      canGenerate = false;
      warnings.push("大模型配置尚未通过接口连接、模型响应、结构化输出和本人识别四项测试。");
    }

    const windowRaw = isPlainObject(raw.window) ? raw.window : {};
    const effectiveReportRaw = isPlainObject(raw.effective_report) ? raw.effective_report : {};
    const effectiveFiltersRaw = isPlainObject(raw.effective_filters) ? raw.effective_filters : config.filters;
    const instructionRaw = isPlainObject(raw.instruction_interpretation) ? raw.instruction_interpretation : {};
    return {
      resolved,
      start: firstString(windowRaw.start, raw.start, config.start),
      end: firstString(windowRaw.end, raw.end, config.end),
      timezone: firstString(windowRaw.timezone, raw.timezone, config.timezone),
      warnings: uniqueStrings(warnings),
      canGenerate,
      effectiveFilters: effectiveFiltersRaw,
      effectiveReport: {
        subjectName: firstString(effectiveReportRaw.subject_name, config.report.subject_name),
        onlySubjectItems: typeof effectiveReportRaw.only_subject_items === "boolean"
          ? effectiveReportRaw.only_subject_items
          : config.filters.only_related_to_me === true,
        title: firstString(effectiveReportRaw.title, config.report.title),
        grouping: firstString(effectiveReportRaw.grouping, config.report.grouping),
        maxLengthChars: Number(effectiveReportRaw.max_length_chars || config.report.max_length_chars),
        includeFullReviewAppendix: false,
        sections: arrayValue(effectiveReportRaw.sections).length ? arrayValue(effectiveReportRaw.sections) : config.report.sections,
        selectedCategories: arrayValue(effectiveReportRaw.selected_explicit_categories),
      },
      instructionInterpretation: {
        applied: arrayValue(instructionRaw.applied),
        conflicts: arrayValue(instructionRaw.conflicts),
        unsupported: arrayValue(instructionRaw.unsupported),
      },
    };
  }

  function renderPreview(preview, config) {
    el.previewPlaceholder.hidden = true;
    el.previewResult.hidden = false;
    el.previewResult.classList.toggle("is-blocked", !preview.canGenerate);
    el.previewStatusIcon.className = `status-icon ${preview.canGenerate ? "status-icon-ok" : "status-icon-error"}`;
    el.previewStatusIcon.textContent = preview.canGenerate ? "✓" : "!";
    el.previewStatusTitle.textContent = preview.canGenerate ? "范围可以生成" : "当前范围不能生成";
    el.previewStatusCopy.textContent = preview.canGenerate ? "仍需勾选明确确认" : "请先修正未解决的问题";
    el.previewStamp.textContent = `预览于 ${formatDateTime(new Date().toISOString(), true)}`;
    el.previewConversations.textContent = preview.resolved.length
      ? preview.resolved.map((item) => item.title).join("、")
      : "没有确认到可用会话";
    el.previewWindow.textContent = `${displayLocalDateTime(preview.start)} — ${displayLocalDateTime(preview.end)}（${preview.timezone}）`;
    el.previewFilters.textContent = describeFilters(preview.effectiveFilters);
    const effectiveReport = preview.effectiveReport;
    const appliedCount = preview.instructionInterpretation.applied.length;
    const categoryCopy = effectiveReport.selectedCategories.length
      ? `；实际事实分类 ${effectiveReport.selectedCategories.length} 类`
      : "";
    const modelTestCopy = state.llmTestPassed ? "大模型测试已通过" : "大模型测试未通过";
    const reviewCopy = "报告不展示原文、消息出处或全量消息清单";
    const subjectCopy = effectiveReport.onlySubjectItems
      ? `报告本人“${effectiveReport.subjectName}”，AI 摘要只汇总本人事项`
      : `报告本人“${effectiveReport.subjectName}”，AI 摘要保留团队事项并区分参与人`;
    el.previewReport.textContent = `${effectiveReport.title}；${subjectCopy}；输出“本周已完成事项”和“周报”两份；${groupingLabel(effectiveReport.grouping)}；AI 摘要最多 ${effectiveReport.maxLengthChars} 字符；${reviewCopy}${categoryCopy}；本次补充要求 ${appliedCount} 条；${modelTestCopy}`;
    el.previewWarningsTitle.textContent = preview.canGenerate ? "需要你确认的不确定项" : "必须先修正的问题";
    renderWarningList(el.previewWarnings, el.previewWarningList, preview.warnings);
  }

  function renderValidationErrors(errors) {
    el.validationSummary.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = "请先完成以下设置";
    const list = document.createElement("ul");
    errors.forEach((error) => {
      const item = document.createElement("li");
      item.textContent = error;
      list.append(item);
    });
    el.validationSummary.append(strong, list);
    el.validationSummary.hidden = false;
    el.validationSummary.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function focusFirstInvalid(config) {
    if (!config.conversation_ids.length) {
      el.conversationSearch.focus();
    } else if (!config.start) {
      el.startTime.focus();
    } else if (!config.end || config.start >= config.end) {
      el.endTime.focus();
    } else if (!config.report.subject_name) {
      el.reportSubjectName.focus();
    } else if (!config.report.custom_instruction) {
      el.customInstruction.focus();
    }
  }

  function updateGenerateAvailability() {
    const currentSignature = stableStringify(buildConfiguration());
    const previewCurrent = Boolean(state.preview && state.previewSignature === currentSignature);
    const uncertaintyAccepted = el.uncertaintyAckWrap.hidden || el.uncertaintyAck.checked;
    el.generateButton.disabled = !(
      previewCurrent &&
      state.preview.canGenerate &&
      state.llmTestPassed &&
      el.explicitConfirm.checked &&
      uncertaintyAccepted &&
      !state.activeJobId
    );
  }

  async function createJob() {
    const config = buildConfiguration();
    const errors = validateConfiguration(config);
    const signature = stableStringify(config);
    if (errors.length || !state.preview || state.previewSignature !== signature) {
      invalidatePreview();
      renderValidationErrors(errors.length ? errors : ["配置已变化，请重新预览并确认范围。"]);
      return;
    }
    if (!state.llmTestPassed) {
      showToast("模型尚未通过测试", "请先保存模型配置，并让接口连接、模型响应、结构化输出和本人识别四项测试全部通过。", "warning", 9000);
      return;
    }
    if (!state.preview.canGenerate || !el.explicitConfirm.checked || (!el.uncertaintyAckWrap.hidden && !el.uncertaintyAck.checked)) {
      showToast("尚未确认", "请核对范围并完成明确确认。", "warning");
      return;
    }

    setButtonLoading(el.generateButton, true, "正在创建任务");
    try {
      const raw = await apiFetch("/jobs", {
        method: "POST",
        body: JSON.stringify({
          ...config,
          explicit_confirm: true,
          uncertainty_ack: Boolean(state.preview.warnings.length && el.uncertaintyAck.checked),
        }),
      });
      const id = firstString(raw.id, raw.job_id);
      if (!id) throw new Error("服务端没有返回任务 ID，无法安全跟踪本次生成。");
      state.activeJobId = id;
      state.activeJob = normalizeJob({ ...raw, id });
      state.reportDocuments = {};
      state.reportMarkdown = "";
      state.reportJobId = null;
      el.reportPreviewPanel.hidden = true;
      el.explicitConfirm.checked = false;
      el.generateButton.disabled = true;
      showProcessingPanel(state.activeJob);
      renderJob(state.activeJob);
      showToast("已开始手动生成", "系统将先处理钉钉数据，再进行分析和报告生成。", "success");
      scheduleJobPoll(200);
    } catch (error) {
      showToast("无法开始生成", error.message, "error", 9000);
    } finally {
      setButtonLoading(el.generateButton, false);
      updateGenerateAvailability();
    }
  }

  function showProcessingPanel(job) {
    el.processingPanel.hidden = false;
    el.activeJobId.textContent = `任务 ${job.id}`;
    el.processingPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function scheduleJobPoll(delay) {
    window.clearTimeout(state.pollTimer);
    if (!state.activeJobId) return;
    state.pollTimer = window.setTimeout(pollActiveJob, delay == null ? POLL_INTERVAL_MS : delay);
  }

  async function pollActiveJob() {
    if (!state.activeJobId) return;
    try {
      const raw = await apiFetch(`/jobs/${encodeURIComponent(state.activeJobId)}`);
      const job = normalizeJob(raw, state.activeJobId);
      state.activeJob = job;
      renderJob(job);

      if (TERMINAL_SUCCESS.has(job.status)) {
        const completedId = state.activeJobId;
        state.activeJobId = null;
        window.clearTimeout(state.pollTimer);
        await loadReport(completedId, job);
        await loadHistory();
        updateGenerateAvailability();
        showToast("报告已生成", "已完成数据处理、分析和报告生成。", "success");
      } else if (TERMINAL_FAILURE.has(job.status)) {
        state.activeJobId = null;
        window.clearTimeout(state.pollTimer);
        updateGenerateAvailability();
        showToast("报告生成失败", job.error || "请查看数据处理结果中的错误信息。", "error", 10000);
        loadHistory();
      } else if (NEEDS_ATTENTION.has(job.status)) {
        state.activeJobId = null;
        window.clearTimeout(state.pollTimer);
        updateGenerateAvailability();
        showToast("任务需要人工确认", "系统发现不确定的数据情况并已停止继续生成，请查看详情。", "warning", 10000);
        loadHistory();
      } else {
        scheduleJobPoll();
      }
    } catch (error) {
      el.processingSubtitle.textContent = `暂时无法获取任务进度：${error.message}`;
      scheduleJobPoll(3500);
    }
  }

  function normalizeJob(raw, fallbackId) {
    const status = firstString(raw.status, raw.state, "queued").toLowerCase();
    const stage = normalizeStage(firstString(raw.stage, raw.phase, status));
    let progress = Number(raw.progress ?? raw.percent ?? stageProgress(stage));
    if (!Number.isFinite(progress)) progress = stageProgress(stage);
    if (progress > 0 && progress <= 1) progress *= 100;
    progress = Math.max(0, Math.min(100, progress));
    if (TERMINAL_SUCCESS.has(status)) progress = 100;

    const warnings = normalizeWarnings(raw.warnings || raw.uncertainties || raw.warning);
    const failed = arrayFrom(raw, ["failed_conversations", "unavailable_conversations"]);
    failed.forEach((item) => {
      const title = typeof item === "string" ? item : firstString(item.title, item.name, item.id, "未知会话");
      const reason = typeof item === "string" ? "读取失败" : firstString(item.reason, item.error, item.message, "读取失败");
      warnings.push(`${title}：${reason}`);
    });

    return {
      id: firstString(raw.id, raw.job_id, fallbackId, "未知"),
      status,
      stage,
      progress,
      error: errorMessage(
        { error: raw.error, detail: raw.detail, message: TERMINAL_FAILURE.has(status) ? raw.message : "" },
        ""
      ),
      counts: isPlainObject(raw.counts) ? raw.counts : {},
      warnings: uniqueStrings(warnings),
      reportUrl: firstString(raw.report_url, raw.report && raw.report.url, ""),
      documents: normalizeReportDocuments(raw.documents || raw.available_documents || (raw.report && raw.report.documents)),
      integrity: isPlainObject(raw.integrity) ? raw.integrity : {},
      title: firstString(raw.title, raw.report_title, raw.config && raw.config.report && raw.config.report.title, "钉钉聊天报告"),
      createdAt: firstString(raw.created_at, raw.created, ""),
    };
  }

  function normalizeStage(stage) {
    const value = String(stage || "").toLowerCase();
    if (["queued", "pending", "created", "preparing", "scope"].includes(value)) return "queued";
    if (["collecting", "reading", "fetching", "extracting", "data_collection"].includes(value)) return "collecting";
    if (["cleaning", "validating", "processing", "deduplicating", "stabilizing", "data_processing"].includes(value)) return "cleaning";
    if (["analyzing", "analysis", "summarizing"].includes(value)) return "analyzing";
    if (["reporting", "generating", "rendering", "writing"].includes(value)) return "reporting";
    if (TERMINAL_SUCCESS.has(value)) return "complete";
    return "queued";
  }

  function stageProgress(stage) {
    return { queued: 5, collecting: 26, cleaning: 52, analyzing: 73, reporting: 90, complete: 100 }[stage] || 5;
  }

  function renderJob(job) {
    el.processingPanel.hidden = false;
    el.activeJobId.textContent = `任务 ${job.id}`;
    el.progressBar.style.width = `${job.progress}%`;
    el.progressTrack.setAttribute("aria-valuenow", String(Math.round(job.progress)));
    el.processingSubtitle.textContent = jobStatusCopy(job);

    const order = ["queued", "collecting", "cleaning", "analyzing", "reporting"];
    const currentIndex = job.stage === "complete" ? order.length : Math.max(0, order.indexOf(job.stage));
    el.pipelineSteps.querySelectorAll("li").forEach((item, index) => {
      item.classList.toggle("is-complete", index < currentIndex || job.stage === "complete");
      item.classList.toggle("is-current", index === currentIndex && job.stage !== "complete" && !TERMINAL_FAILURE.has(job.status) && !NEEDS_ATTENTION.has(job.status));
      const marker = item.querySelector(":scope > span");
      marker.textContent = index < currentIndex || job.stage === "complete" ? "✓" : String(index + 1);
    });

    renderCounts(job.counts);
    renderWarningList(el.jobWarnings, el.jobWarningList, job.warnings);
    if (job.error) {
      el.jobError.textContent = job.error;
      el.jobError.hidden = false;
    } else if (NEEDS_ATTENTION.has(job.status)) {
      el.jobError.textContent = "发现无法自动确定的数据情况，任务已停止继续生成。请查看警告并调整范围后重新预览。";
      el.jobError.hidden = false;
    } else {
      el.jobError.hidden = true;
    }
  }

  function jobStatusCopy(job) {
    if (TERMINAL_SUCCESS.has(job.status)) return "数据处理和报告生成已完成。";
    if (TERMINAL_FAILURE.has(job.status)) return "任务失败，未将失败或空结果当成完整数据。";
    if (NEEDS_ATTENTION.has(job.status)) return "任务因不确定项暂停，需要人工检查。";
    return {
      queued: "正在固定本次会话、时间和配置范围…",
      collecting: "正在逐个读取选定会话的消息…",
      cleaning: "正在过滤、去重并检查数据稳定性…",
      analyzing: "数据处理完成，正在分析事实、决策、风险和待办…",
      reporting: "正在按照本次要求组织报告与证据…",
    }[job.stage] || "正在执行本次手动任务…";
  }

  function renderCounts(counts) {
    const definitions = [
      [["conversations_requested", "requested_conversations", "selected_conversations", "选定会话"], "选定会话"],
      [["conversations_processed", "successful_conversations", "processed_conversations", "成功读取"], "成功读取"],
      [["conversations_failed", "failed_conversations", "读取失败"], "读取失败"],
      [["raw_messages", "messages_raw", "total_messages", "原始消息", "原始快照消息"], "原始消息"],
      [["duplicates", "duplicate_messages", "messages_deduplicated", "重复消息"], "重复消息"],
      [["out_of_range", "messages_out_of_range", "范围外"], "范围外"],
      [["excluded_messages", "filtered_messages", "messages_excluded", "筛选排除", "被筛选排除"], "筛选排除"],
      [["analysis_messages", "messages_analyzed", "included_messages", "进入分析"], "进入分析"],
      [["review_messages", "manual_review_messages", "人工核对消息"], "人工核对消息"],
      [["edited_messages", "edits", "编辑消息"], "编辑消息"],
      [["recalled_messages", "recalls", "撤回标记"], "撤回标记"],
      [["evidence_findings", "findings", "证据结论"], "证据结论"],
    ];
    const entries = definitions.map(([keys, label]) => {
      const key = keys.find((candidate) => Object.prototype.hasOwnProperty.call(counts, candidate));
      return key ? { label, value: counts[key] } : null;
    }).filter(Boolean);

    el.countsGrid.replaceChildren();
    if (!entries.length) {
      el.countsGrid.hidden = true;
      return;
    }
    const fragment = document.createDocumentFragment();
    entries.forEach((entry) => {
      const item = document.createElement("dl");
      item.className = "count-item";
      const term = document.createElement("dt");
      term.textContent = entry.label;
      const value = document.createElement("dd");
      value.textContent = formatNumber(entry.value);
      item.append(term, value);
      fragment.append(item);
    });
    el.countsGrid.append(fragment);
    el.countsGrid.hidden = false;
  }

  async function loadReport(jobId, job) {
    const documentTypes = Object.keys(REPORT_DOCUMENTS);
    try {
      const results = await Promise.allSettled(documentTypes.map((documentType) => (
        loadReportDocument(jobId, documentType, job)
      )));
      const documents = {};
      const failures = [];
      results.forEach((result, index) => {
        const documentType = documentTypes[index];
        if (result.status === "fulfilled" && result.value) documents[documentType] = result.value;
        else failures.push(`${REPORT_DOCUMENTS[documentType].title}：${result.reason && result.reason.message ? result.reason.message : "未返回内容"}`);
      });

      if (!Object.keys(documents).length) throw new Error(failures.join("；") || "两份结果接口都没有返回 Markdown 内容。");

      state.reportDocuments = documents;
      state.reportJobId = jobId;
      state.reportBaseTitle = job && job.title ? job.title : "钉钉聊天工作报告";
      el.reportPreviewTitle.textContent = `${state.reportBaseTitle} · 两份结果`;
      el.reportIntegritySummary.textContent = describeIntegrity(job && job.integrity);
      updateReportDocumentTabs();
      const preferred = documents.completed_items ? "completed_items" : "weekly_report";
      switchReportDocument(preferred);
      el.reportPreviewPanel.hidden = false;
      el.reportPreviewPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      if (failures.length) showToast("部分结果无法打开", failures.join("；"), "warning", 10000);
    } catch (error) {
      showToast("无法打开生成结果", error.message, "error", 8500);
    }
  }

  async function loadReportDocument(jobId, documentType, job) {
    const url = safeReportPath(jobId, documentType);
    try {
      const raw = await apiFetch(url);
      const markdown = reportMarkdownFromPayload(raw, documentType);
      if (!markdown) throw new Error("接口没有返回 Markdown 内容。");
      return markdown;
    } catch (error) {
      if (
        documentType !== "completed_items" ||
        !job ||
        !job.reportUrl ||
        !Array.isArray(job.documents) ||
        !job.documents.includes("completed_items")
      ) throw error;
      const raw = await apiFetch(job.reportUrl);
      const markdown = reportMarkdownFromPayload(raw, documentType);
      if (!markdown) throw error;
      return markdown;
    }
  }

  function reportMarkdownFromPayload(raw, documentType) {
    if (typeof raw === "string") return raw;
    const documents = isPlainObject(raw.documents) ? raw.documents : {};
    const selected = documents[documentType];
    if (typeof selected === "string") return selected;
    if (isPlainObject(selected)) return firstString(selected.markdown, selected.content, selected.report, "");
    return firstString(raw.markdown, raw.content, typeof raw.report === "string" ? raw.report : "", raw.message, "");
  }

  function safeReportPath(jobId, documentType) {
    return `/api/jobs/${encodeURIComponent(jobId)}/report?document=${encodeURIComponent(documentType)}`;
  }

  function normalizeReportDocuments(value) {
    if (Array.isArray(value)) {
      return uniqueStrings(value.map((item) => (
        typeof item === "string" ? item : firstString(item.type, item.document, item.name, item.id, "")
      )).filter((item) => Object.prototype.hasOwnProperty.call(REPORT_DOCUMENTS, item)));
    }
    if (isPlainObject(value)) {
      return Object.keys(value).filter((item) => Object.prototype.hasOwnProperty.call(REPORT_DOCUMENTS, item));
    }
    return [];
  }

  function updateReportDocumentTabs() {
    Object.keys(REPORT_DOCUMENTS).forEach((documentType) => {
      const button = document.querySelector(`[data-report-document="${documentType}"]`);
      const available = Boolean(state.reportDocuments[documentType]);
      button.disabled = !available;
      button.title = available ? `查看${REPORT_DOCUMENTS[documentType].title}` : `${REPORT_DOCUMENTS[documentType].title}暂不可用`;
    });
  }

  function switchReportDocument(documentType) {
    if (!REPORT_DOCUMENTS[documentType] || !state.reportDocuments[documentType]) return;
    state.activeReportDocument = documentType;
    state.reportMarkdown = state.reportDocuments[documentType];
    Object.keys(REPORT_DOCUMENTS).forEach((candidate) => {
      const button = document.querySelector(`[data-report-document="${candidate}"]`);
      const active = candidate === documentType;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
    });
    const meta = REPORT_DOCUMENTS[documentType];
    el.markdownPreview.setAttribute("aria-labelledby", meta.tabId);
    el.reportDocumentDescription.textContent = `正在预览：${meta.title}（${meta.description}）`;
    renderMarkdown(state.reportMarkdown);
  }

  function describeIntegrity(integrity) {
    if (!integrity || !Object.keys(integrity).length) return "报告已绑定本次会话、时间和数据快照。";
    const complete = integrity.data_complete === true;
    const sealed = integrity.sealed === true;
    const status = firstString(integrity.status, "");
    if (complete && sealed) return `数据快照已封存${status ? `（${status}）` : ""}。`;
    return `报告已生成，但数据完整性仍需注意${status ? `（${status}）` : ""}；请查看上方警告。`;
  }

  function renderMarkdown(markdown) {
    el.markdownPreview.replaceChildren();
    const fragment = document.createDocumentFragment();
    const lines = String(markdown).replace(/\r\n?/g, "\n").split("\n");
    let list = null;
    let listType = null;
    let paragraph = [];
    let inCode = false;
    let codeLines = [];

    const flushParagraph = () => {
      if (!paragraph.length) return;
      const p = document.createElement("p");
      p.append(...inlineNodes(paragraph.join(" ")));
      fragment.append(p);
      paragraph = [];
    };
    const flushList = () => {
      if (list) fragment.append(list);
      list = null;
      listType = null;
    };

    lines.forEach((line) => {
      if (/^```/.test(line)) {
        flushParagraph();
        flushList();
        if (inCode) {
          const pre = document.createElement("pre");
          const code = document.createElement("code");
          code.textContent = codeLines.join("\n");
          pre.append(code);
          fragment.append(pre);
          codeLines = [];
          inCode = false;
        } else {
          inCode = true;
        }
        return;
      }
      if (inCode) {
        codeLines.push(line);
        return;
      }

      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
      const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
      const quote = line.match(/^>\s?(.*)$/);

      if (heading) {
        flushParagraph();
        flushList();
        const h = document.createElement(`h${heading[1].length}`);
        h.append(...inlineNodes(heading[2]));
        fragment.append(h);
      } else if (unordered || ordered) {
        flushParagraph();
        const type = ordered ? "ol" : "ul";
        if (!list || listType !== type) {
          flushList();
          list = document.createElement(type);
          listType = type;
        }
        const item = document.createElement("li");
        item.append(...inlineNodes((unordered || ordered)[1]));
        list.append(item);
      } else if (quote) {
        flushParagraph();
        flushList();
        const blockquote = document.createElement("blockquote");
        blockquote.append(...inlineNodes(quote[1]));
        fragment.append(blockquote);
      } else if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
        flushParagraph();
        flushList();
        fragment.append(document.createElement("hr"));
      } else if (!line.trim()) {
        flushParagraph();
        flushList();
      } else {
        paragraph.push(line.trim());
      }
    });

    if (inCode && codeLines.length) {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = codeLines.join("\n");
      pre.append(code);
      fragment.append(pre);
    }
    flushParagraph();
    flushList();
    el.markdownPreview.append(fragment);
  }

  function inlineNodes(text) {
    const fragment = document.createDocumentFragment();
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let cursor = 0;
    for (const match of String(text).matchAll(pattern)) {
      if (match.index > cursor) fragment.append(document.createTextNode(text.slice(cursor, match.index)));
      const token = match[0];
      if (token.startsWith("**")) {
        const strong = document.createElement("strong");
        strong.textContent = token.slice(2, -2);
        fragment.append(strong);
      } else {
        const code = document.createElement("code");
        code.textContent = token.slice(1, -1);
        fragment.append(code);
      }
      cursor = match.index + token.length;
    }
    if (cursor < text.length) fragment.append(document.createTextNode(text.slice(cursor)));
    return Array.from(fragment.childNodes);
  }

  async function copyReport() {
    if (!state.reportMarkdown) return;
    const documentMeta = REPORT_DOCUMENTS[state.activeReportDocument];
    try {
      await navigator.clipboard.writeText(state.reportMarkdown);
      showToast("已复制", `${documentMeta ? documentMeta.title : "Markdown 结果"}已复制到剪贴板。`, "success");
    } catch (_) {
      const textarea = document.createElement("textarea");
      textarea.value = state.reportMarkdown;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      showToast("已复制", `${documentMeta ? documentMeta.title : "Markdown 结果"}已复制到剪贴板。`, "success");
    }
  }

  async function downloadReport() {
    if (!state.reportMarkdown) return;
    const meta = REPORT_DOCUMENTS[state.activeReportDocument] || REPORT_DOCUMENTS.completed_items;
    let blob;
    let serverFilename = "";
    if (state.reportJobId) {
      try {
        const response = await fetch(
          `/api/jobs/${encodeURIComponent(state.reportJobId)}/download?document=${encodeURIComponent(state.activeReportDocument)}`,
          { credentials: "same-origin" },
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        blob = await response.blob();
        const disposition = response.headers.get("content-disposition") || "";
        const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
        const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
        if (utf8Match) serverFilename = decodeURIComponent(utf8Match[1]);
        else if (plainMatch) serverFilename = plainMatch[1];
      } catch (_) {
        blob = null;
      }
    }
    if (!blob) blob = new Blob([state.reportMarkdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = serverFilename || `${safeFilename(`${state.reportBaseTitle}-${meta.filenameSuffix}`)}.md`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast("已开始下载", `${meta.title}已按独立文件下载。`, "success");
  }

  async function loadHistory(showFeedback) {
    if (showFeedback) el.historyList.innerHTML = '<div class="empty-history">正在刷新…</div>';
    try {
      const raw = await apiFetch("/jobs");
      const jobs = arrayFrom(raw, ["items", "jobs", "results", "data"]).map((item) => normalizeJob(item));
      renderHistory(jobs);
      if (showFeedback) showToast("历史记录已更新", `共 ${jobs.length} 条。`, "success");
    } catch (error) {
      el.historyList.replaceChildren();
      const empty = document.createElement("div");
      empty.className = "empty-history";
      empty.textContent = "暂时无法读取历史记录";
      el.historyList.append(empty);
      if (showFeedback) showToast("刷新历史失败", error.message, "error");
    }
  }

  function renderHistory(jobs) {
    el.historyList.replaceChildren();
    if (!jobs.length) {
      const empty = document.createElement("div");
      empty.className = "empty-history";
      empty.textContent = "还没有手动生成记录";
      el.historyList.append(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    jobs.slice(0, 8).forEach((job) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item";
      const top = document.createElement("span");
      top.className = "history-item-top";
      const title = document.createElement("strong");
      title.textContent = job.title;
      const badge = document.createElement("span");
      const statusClass = TERMINAL_SUCCESS.has(job.status) ? "complete" : TERMINAL_FAILURE.has(job.status) ? "failed" : "";
      badge.className = `history-status ${statusClass}`;
      badge.textContent = historyStatusLabel(job.status);
      const meta = document.createElement("small");
      meta.textContent = `${formatDateTime(job.createdAt || "")} · ${job.id}`;
      top.append(title, badge);
      button.append(top, meta);
      button.addEventListener("click", () => openHistoryJob(job));
      fragment.append(button);
    });
    el.historyList.append(fragment);
  }

  async function openHistoryJob(job) {
    if (TERMINAL_SUCCESS.has(job.status)) {
      await loadReport(job.id, job);
      return;
    }
    try {
      const raw = await apiFetch(`/jobs/${encodeURIComponent(job.id)}`);
      const freshJob = normalizeJob(raw, job.id);
      state.activeJob = freshJob;
      showProcessingPanel(freshJob);
      renderJob(freshJob);
      if (!TERMINAL_FAILURE.has(freshJob.status) && !NEEDS_ATTENTION.has(freshJob.status) && !TERMINAL_SUCCESS.has(freshJob.status)) {
        state.activeJobId = freshJob.id;
        scheduleJobPoll();
      } else if (TERMINAL_SUCCESS.has(freshJob.status)) {
        await loadReport(freshJob.id, freshJob);
      }
    } catch (error) {
      showToast("无法打开任务", error.message, "error");
    }
  }

  function historyStatusLabel(status) {
    if (TERMINAL_SUCCESS.has(status)) return "已完成";
    if (TERMINAL_FAILURE.has(status)) return "失败";
    if (NEEDS_ATTENTION.has(status)) return "待确认";
    return "处理中";
  }

  function saveConfigurationSoon() {
    el.autosaveState.textContent = "正在保存…";
    window.clearTimeout(state.saveTimer);
    state.saveTimer = window.setTimeout(saveConfiguration, 180);
  }

  function saveConfiguration() {
    try {
      const payload = {
        version: 2,
        saved_at: new Date().toISOString(),
        data_mode: state.status?.dataMode || state.restoredDataMode || null,
        config: buildConfiguration(),
        selected_meta: Array.from(state.selectedMeta.values()).map((item) => ({
          id: item.id,
          title: item.title,
          kind: item.kind,
          participants: item.participants || [],
          available: item.available !== false,
          source: item.source || "",
          warning: item.warning || "",
        })),
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      el.autosaveState.textContent = "已自动保存";
    } catch (_) {
      el.autosaveState.textContent = "无法本地保存";
    }
  }

  function restoreConfiguration() {
    try {
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!stored || stored.version !== 2 || !isPlainObject(stored.config)) return;
      const config = stored.config;
      const filters = isPlainObject(config.filters) ? config.filters : {};
      const report = isPlainObject(config.report) ? config.report : {};
      state.restoredDataMode = firstString(stored.data_mode, "") || null;

      (stored.selected_meta || []).map(normalizeConversation).filter((item) => item.id).forEach((item) => state.selectedMeta.set(item.id, item));
      (config.conversation_ids || []).forEach((id) => state.selectedIds.add(String(id)));
      el.startTime.value = datetimeLocalValue(config.start);
      el.endTime.value = datetimeLocalValue(config.end);
      setValue(el.timezone, config.timezone);
      setCheckedValues('input[name="messageType"]', filters.message_types);
      el.includeSenders.value = joinList(filters.include_senders);
      el.excludeSenders.value = joinList(filters.exclude_senders);
      el.includeKeywords.value = joinList(filters.include_keywords);
      el.excludeKeywords.value = joinList(filters.exclude_keywords);
      setChecked(el.includeSystem, filters.include_system_messages);
      setChecked(el.includeBots, filters.include_bot_messages);
      setChecked(el.includeRecalled, filters.include_recalled ?? filters.include_recalled_markers, true);
      setChecked(el.expandReplies, false);
      setChecked(el.expandForwards, false);
      setChecked(el.onlyRelatedToMe, filters.only_related_to_me, true);
      if (report.subject_name != null) el.reportSubjectName.value = String(report.subject_name);
      setValue(el.casualFilter, filters.casual_filter ?? filters.casual_chat_filter);
      if (report.title != null) el.reportTitle.value = String(report.title);
      setValue(el.reportGrouping, report.grouping);
      setValue(el.reportLength, report.length);
      setValue(el.reportLanguage, "zh-CN");
      setValue(el.inferencePolicy, "facts_only");
      setCheckedValues('input[name="reportSection"]', report.sections);
      if (report.custom_instruction != null) {
        const restoredInstruction = String(report.custom_instruction)
          .replace(/(?:关键|重要)?结论附消息出处[。；;]?/g, "")
          .trim();
        el.customInstruction.value = restoredInstruction || "只写事实。不明确信息标记为待确认。不要编造负责人。不要编造截止时间。不要编造结论。";
      }
      setChecked(el.evidenceEnabled, true);
      setValue(el.evidenceLevel, "all_items");
      setChecked(el.includeFullReviewAppendix, false);
    } catch (_) {
      // Invalid or blocked local storage is ignored; defaults remain in place.
    }
  }

  function reconcileRestoredDataMode(currentMode) {
    const current = modeFamily(currentMode);
    if (!current) return;
    const restored = modeFamily(state.restoredDataMode);
    const sources = Array.from(state.selectedMeta.values()).map((item) => String(item.source || "").toLowerCase());
    const legacyMismatch = !restored && sources.some((source) => (
      (current === "real" && /fixture|test|demo|mock/.test(source))
      || (current === "fixture" && /dws|real|live|production/.test(source))
    ));
    const changed = Boolean(restored && restored !== current);
    state.restoredDataMode = current;
    if (!changed && !legacyMismatch) return;

    state.selectedIds.clear();
    state.selectedMeta.clear();
    invalidatePreview();
    renderConversations();
    renderSelectedConversations();
    updateConfigurationSummary();
    saveConfigurationSoon();
    showToast(
      "已隔离测试与真实会话",
      "检测到数据模式变化，已清除上一个模式保存的会话选择；通用时间和报告设置仍保留。",
      "warning",
      8500,
    );
  }

  function modeFamily(value) {
    const mode = String(value || "").toLowerCase();
    if (/fixture|test|demo|mock/.test(mode)) return "fixture";
    if (/real|live|production/.test(mode)) return "real";
    return "";
  }

  function updateConfigurationSummary() {
    const config = buildConfiguration();
    const selected = Array.from(state.selectedIds).map((id) => state.selectedMeta.get(id)?.title || id);
    el.summaryConversations.textContent = selected.length ? `${selected.length} 个：${selected.slice(0, 3).join("、")}${selected.length > 3 ? "…" : ""}` : "尚未选择";
    el.summaryStart.textContent = config.start ? `${displayLocalDateTime(config.start)} · ${config.timezone}` : "—";
    el.summaryEnd.textContent = config.end ? `${displayLocalDateTime(config.end)} · ${config.timezone}` : "—";
    el.summaryTypes.textContent = config.filters.message_types.length ? config.filters.message_types.map(messageTypeLabel).join("、") : "尚未选择";
    el.summarySections.textContent = config.report.sections.length
      ? `本人：${config.report.subject_name || "未填写"} · ${config.filters.only_related_to_me ? "仅汇总本人" : "区分团队事项"} · ${config.report.sections.length} 个章节 · ${groupingLabel(config.report.grouping)} · 不展示原文出处`
      : "尚未选择";
  }

  function updateInstructionCounter() {
    el.instructionCount.textContent = String(el.customInstruction.value.length);
  }

  function describeFilters(filters) {
    const messageTypes = arrayValue(filters.message_types);
    const details = [messageTypes.map(messageTypeLabel).join("、") || "无消息类型"];
    if (!filters.include_system_messages) details.push("排除系统通知");
    if (!filters.include_bot_messages) details.push("排除机器人");
    const includeKeywords = arrayValue(filters.include_keywords);
    const excludeKeywords = arrayValue(filters.exclude_keywords);
    if (includeKeywords.length) details.push(`关键词：${includeKeywords.join("/")}`);
    if (excludeKeywords.length) details.push(`排除：${excludeKeywords.join("/")}`);
    if (filters.only_related_to_me) details.push("AI 摘要仅汇总报告本人（全量核对不缩减）");
    if (filters.exclude_casual === true || filters.casual_filter === "conservative") details.push("保守过滤闲聊");
    else details.push("保留普通闲聊");
    return details.join("；");
  }

  function groupingLabel(value) {
    return { topic: "按主题", conversation: "按会话", date: "按日期", time: "按日期", owner: "按负责人" }[value] || value || "按主题";
  }

  function messageTypeLabel(value) {
    return { text: "文本", file: "文件", image: "图片", link: "链接", audio: "语音", video: "视频" }[value] || value;
  }

  function abbreviateId(value) {
    const text = String(value);
    if (text.length <= 18) return text;
    return `${text.slice(0, 9)}…${text.slice(-6)}`;
  }

  function reportLengthCharacters(value) {
    return { brief: 5000, standard: 15000, detailed: 50000 }[value] || 15000;
  }

  function renderWarningList(container, list, warnings) {
    list.replaceChildren();
    container.hidden = !warnings.length;
    if (!warnings.length) return;
    const fragment = document.createDocumentFragment();
    warnings.forEach((warning) => {
      const item = document.createElement("li");
      item.textContent = warning;
      fragment.append(item);
    });
    list.append(fragment);
  }

  function normalizeWarnings(value) {
    if (value == null || value === "") return [];
    const items = Array.isArray(value) ? value : [value];
    return items.map((item) => {
      if (typeof item === "string") return item;
      if (!isPlainObject(item)) return String(item);
      const conversation = firstString(item.conversation_title, item.conversation_name, item.title, item.id, "");
      const message = firstString(item.message, item.reason, item.warning, item.detail, JSON.stringify(item));
      return conversation && !message.includes(conversation) ? `${conversation}：${message}` : message;
    }).filter(Boolean);
  }

  function makeLoadingState(title, copy) {
    const wrapper = makeEmptyState(title, copy);
    const spinner = document.createElement("span");
    spinner.className = "loading-spinner";
    spinner.setAttribute("aria-hidden", "true");
    wrapper.prepend(spinner);
    return wrapper;
  }

  function makeEmptyState(title, copy) {
    const wrapper = document.createElement("div");
    wrapper.className = "empty-state compact-empty";
    const strong = document.createElement("strong");
    strong.textContent = title;
    const p = document.createElement("p");
    p.textContent = copy;
    wrapper.append(strong, p);
    return wrapper;
  }

  function setButtonLoading(button, loading, loadingLabel) {
    if (!button.dataset.originalLabel) button.dataset.originalLabel = button.textContent.trim();
    button.classList.toggle("is-loading", loading);
    button.disabled = loading;
    if (loading) {
      button.textContent = loadingLabel || "处理中";
      button.setAttribute("aria-busy", "true");
    } else {
      button.textContent = button.dataset.originalLabel;
      button.removeAttribute("aria-busy");
    }
  }

  function showToast(title, message, type, duration) {
    const toast = document.createElement("div");
    toast.className = `toast ${type === "error" ? "is-error" : type === "warning" ? "is-warning" : ""}`;
    toast.setAttribute("role", type === "error" ? "alert" : "status");
    const marker = document.createElement("span");
    marker.textContent = type === "error" ? "!" : type === "warning" ? "△" : "✓";
    const content = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = title;
    const p = document.createElement("p");
    p.textContent = message;
    content.append(strong, p);
    toast.append(marker, content);
    el.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), duration || 4300);
  }

  function checkedValues(selector) {
    return Array.from(document.querySelectorAll(selector)).filter((input) => input.checked).map((input) => input.value);
  }

  function setCheckedValues(selector, values) {
    if (!Array.isArray(values)) return;
    const selected = new Set(values.map(String));
    document.querySelectorAll(selector).forEach((input) => {
      input.checked = selected.has(input.value);
    });
  }

  function setChecked(input, value, fallback) {
    input.checked = typeof value === "boolean" ? value : Boolean(fallback);
  }

  function setValue(input, value) {
    if (value == null || value === "") return;
    const optionExists = !input.options || Array.from(input.options).some((option) => option.value === String(value));
    if (optionExists) input.value = String(value);
  }

  function splitList(value) {
    if (Array.isArray(value)) return uniqueStrings(value.map(String).map((item) => item.trim()).filter(Boolean));
    return uniqueStrings(String(value || "").split(/[，,;；\n]/).map((item) => item.trim()).filter(Boolean));
  }

  function joinList(value) {
    return Array.isArray(value) ? value.join(", ") : firstString(value, "");
  }

  function uniqueStrings(items) {
    return Array.from(new Set(items.map((item) => String(item).trim()).filter(Boolean)));
  }

  function arrayFrom(raw, keys) {
    if (Array.isArray(raw)) return raw;
    for (const key of keys) {
      if (Array.isArray(raw && raw[key])) return raw[key];
      if (isPlainObject(raw && raw[key])) {
        const nested = arrayFrom(raw[key], ["items", "results", "data"]);
        if (nested.length) return nested;
      }
    }
    return [];
  }

  function arrayValue(value) {
    return Array.isArray(value) ? value : [];
  }

  function isPlainObject(value) {
    return value != null && typeof value === "object" && !Array.isArray(value);
  }

  function finiteNumber(...values) {
    for (const value of values) {
      const number = Number(value);
      if (Number.isFinite(number)) return number;
    }
    return 0;
  }

  function nullableFiniteNumber(value, fallback = null) {
    if (value == null || value === "") return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function firstDefinedBoolean(...values) {
    for (const value of values) {
      if (typeof value === "boolean") return value;
      if (value === 1 || value === "1" || value === "true") return true;
      if (value === 0 || value === "0" || value === "false") return false;
    }
    return null;
  }

  function firstString(...values) {
    for (const value of values) {
      if (typeof value === "string" && value.trim()) return value.trim();
      if (typeof value === "number" && Number.isFinite(value)) return String(value);
    }
    return "";
  }

  function errorMessage(payload, fallback) {
    const seen = new Set();
    const extract = (value) => {
      if (typeof value === "string" && value.trim()) return value.trim();
      if (Array.isArray(value)) {
        const messages = value.map(extract).filter(Boolean);
        return uniqueStrings(messages).join("；");
      }
      if (!isPlainObject(value) || seen.has(value)) return "";
      seen.add(value);
      for (const key of ["message", "reason", "detail", "error", "msg", "code"]) {
        const message = extract(value[key]);
        if (message) return message;
      }
      return "";
    };
    return extract(payload) || (fallback !== undefined ? fallback : "请求失败");
  }

  function normalizeLocalDateTime(value) {
    if (!value) return "";
    return value.length === 16 ? `${value}:00` : value;
  }

  function datetimeLocalValue(value) {
    if (!value) return "";
    const match = String(value).match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?/);
    return match ? `${match[1]}T${match[2]}:${match[3] || "00"}` : "";
  }

  function displayLocalDateTime(value) {
    if (!value) return "—";
    const local = String(value).match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?/);
    if (local) return `${local[1]}-${local[2]}-${local[3]} ${local[4]}:${local[5]}${local[6] && local[6] !== "00" ? `:${local[6]}` : ""}`;
    return formatDateTime(value);
  }

  function formatDateTime(value, compact) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    try {
      return new Intl.DateTimeFormat("zh-CN", {
        timeZone: "Asia/Shanghai",
        year: compact ? undefined : "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hourCycle: "h23",
      }).format(date).replace(/\//g, "-");
    } catch (_) {
      return date.toLocaleString("zh-CN");
    }
  }

  function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? new Intl.NumberFormat("zh-CN").format(number) : String(value ?? "—");
  }

  function safeFilename(value) {
    return String(value || "报告").replace(/[\\/:*?"<>|\u0000-\u001f]/g, "_").slice(0, 80) || "报告";
  }

  function stableStringify(value) {
    if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
    if (isPlainObject(value)) {
      return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
    }
    return JSON.stringify(value);
  }
})();

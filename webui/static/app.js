/* YuE2 Console — client. One SSE stream carries engine, run and library news. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var api = function (path, options) {
    return fetch(path, options).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          throw new Error(body.detail || (r.status + " " + r.statusText));
        });
      }
      return r.status === 204 ? null : r.json();
    });
  };

  var STATE = {
    defaults: null,
    writer: null,
    art: null,
    coverPrompt: "",
    settings: null,
    takes: [],
    job: null,          // the run being watched
    take: null,         // the take being inspected
    examples: null,
    peaks: null,
    abcRendered: ""
  };

  var SAMPLER_KNOBS = [
    ["temperature", "Temperature", 0.05],
    ["top_p", "Top-p", 0.01],
    ["top_k", "Top-k", 1],
    ["repetition_penalty", "Repetition penalty", 0.005],
    ["penalty_window", "Penalty window", 1],
    ["max_tokens", "Max tokens", 100]
  ];

  /* ------------------------------------------------------------ helpers */

  function clock(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    var m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function ago(epoch) {
    var d = (Date.now() / 1000) - epoch;
    if (d < 60) return "just now";
    if (d < 3600) return Math.floor(d / 60) + " min ago";
    if (d < 86400) return Math.floor(d / 3600) + " h ago";
    return new Date(epoch * 1000).toLocaleDateString();
  }

  function toast(message, kind) {
    var el = document.createElement("div");
    el.className = "toast" + (kind ? " " + kind : "");
    el.textContent = message;
    $("toasts").appendChild(el);
    setTimeout(function () {
      el.style.transition = "opacity .3s ease";
      el.style.opacity = "0";
      setTimeout(function () { el.remove(); }, 320);
    }, kind === "bad" ? 8000 : 4200);
  }

  /* --------------------------------------------------------------- views */
  /* Compose and the running take share the screen, so the only thing that
     shows and hides is the engine panel. */

  function show(view) {
    if (view === "engine") {
      $("view-engine").classList.toggle("is-hidden");
    } else if (view === "take") {
      $("view-engine").classList.add("is-hidden");
      if (STATE.take) drawWave();
      if (window.innerWidth <= 1150) $("view-take").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  $("engineToggle").addEventListener("click", function () { show("engine"); });

  document.addEventListener("keydown", function (event) {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) return;
    if (event.key === "Escape") $("view-engine").classList.add("is-hidden");
    if (event.key === " " && STATE.take) { event.preventDefault(); togglePlay(); }
  });

  /* ------------------------------------------------------------- engine */

  function paintEngine(engine) {
    var labels = { idle: "Ready", loading: "Loading model", busy: "Generating", error: "Engine error" };
    $("engineLamp").dataset.s = engine.status;
    $("engineState").textContent = labels[engine.status] || engine.status;
    var detail = engine.detail;
    if (!detail) {
      detail = engine.loaded_with
        ? (engine.loaded_with.model.split("/").pop() + " resident")
        : "model not loaded yet";
    }
    $("engineDetail").textContent = detail;
    $("submitNote").textContent = engine.status === "busy"
      ? "A run is in progress; the next song is queued behind it."
      : (engine.loaded_with ? "Model is resident — generation starts immediately."
                            : "Model loads on the first run and stays resident.");
  }

  function paintVram(v) {
    var cell = document.getElementById("vramCell");
    if (!cell) return;
    cell.textContent = v ? (v.used_gib + " / " + v.total_gib + " GiB") : "—";
  }

  function paintHardware(hw) {
    var rows = [
      ["GPU", hw.gpu || "none detected", hw.gpu ? "ok" : "no"],
      ["VRAM", hw.vram_gib ? hw.vram_gib + " GiB" : "—", "", "vramCell"],
      ["CUDA", hw.cuda ? "available" : "unavailable", hw.cuda ? "ok" : "no"],
      ["BF16", hw.bf16 ? "supported" : "no", hw.bf16 ? "ok" : "no"],
      ["torch", hw.torch || "—", ""]
    ];
    var railHtml = "", cardHtml = "";
    rows.forEach(function (row) {
      var idAttr = row[3] ? ' id="' + row[3] + '"' : "";
      railHtml += "<div><dt>" + row[0] + "</dt><dd" + idAttr + ">" + escape(String(row[1])) + "</dd></div>";
      cardHtml += "<div><dt>" + row[0] + "</dt><dd class=\"" + row[2] + "\">" + escape(String(row[1])) + "</dd></div>";
    });
    $("hwStats").innerHTML = railHtml;
    $("hwCard").innerHTML = cardHtml;
  }

  function escape(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* ------------------------------------------------------------ sampling */

  function buildKnobs() {
    ["abc", "semantic"].forEach(function (group) {
      var host = document.querySelector('.knobs[data-group="' + group + '"]');
      host.innerHTML = "";
      SAMPLER_KNOBS.forEach(function (knob) {
        var wrap = document.createElement("label");
        wrap.className = "knob";
        wrap.innerHTML = "<span>" + knob[1] + "</span>";
        var input = document.createElement("input");
        input.type = "number";
        // "any" matters: a stepped number input whose default is not a multiple
        // of the step is invalid, and an invalid control inside a collapsed
        // <details> silently blocks form submission in Chrome.
        input.step = "any";
        input.dataset.group = group;
        input.dataset.key = knob[0];
        input.value = STATE.defaults[group][knob[0]];
        wrap.appendChild(input);
        host.appendChild(wrap);
      });
    });
  }

  function samplingOverrides(group) {
    var out = {};
    Array.prototype.forEach.call(document.querySelectorAll('input[data-group="' + group + '"]'), function (input) {
      var value = parseFloat(input.value);
      if (isFinite(value) && value !== STATE.defaults[group][input.dataset.key]) {
        out[input.dataset.key] = /_(k|window|tokens)$/.test(input.dataset.key) ? Math.round(value) : value;
      }
    });
    return out;
  }

  $("resetSampling").addEventListener("click", function () {
    Array.prototype.forEach.call(document.querySelectorAll("input[data-group]"), function (input) {
      input.value = STATE.defaults[input.dataset.group][input.dataset.key];
    });
    $("cfg").value = "";
    toast("Sampling reset to defaults");
  });

  /* ----------------------------------------------------------------- muse */

  function paintWriter(data) {
    STATE.writer = data;

    // --- llama.cpp -------------------------------------------------------
    var llama = data.llamacpp;
    var llamaReady = llama.installed;
    $("llamaState").textContent = llama.busy ? "working" : (llamaReady ? "installed" : "not installed");
    $("llamaState").dataset.s = llama.busy ? "busy" : (llamaReady ? "ready" : "missing");
    $("llamaInstall").textContent = llamaReady ? "Reinstall llama.cpp" : "Install llama.cpp";
    $("llamaInstall").disabled = !!llama.busy || !llama.supported;
    if (!llama.supported) {
      $("llamaHint").textContent = "The one-click llama.cpp install here covers 64-bit Windows only.";
    }

    $("setLlamaModel").innerHTML = llama.models.length
      ? llama.models.map(function (m) {
          return '<option value="' + escape(m.path) + '">' + escape(m.file) + " · " + m.size_gb +
                 " GiB · " + escape(m.source) + "</option>";
        }).join("")
      : '<option value="">no model found yet</option>';
    if (llama.selected) $("setLlamaModel").value = llama.selected;
    $("writerDirs").textContent = (llama.dirs && llama.dirs.length)
      ? "Also scanning: " + llama.dirs.join("  ·  ")
      : "Scanning the download folder and your Hugging Face cache.";

    $("llamaCatalog").innerHTML = llama.catalog.map(function (entry) {
      var action = entry.installed
        ? '<button type="button" class="btn ghost small" data-drop="' + escape(entry.file) + '">Remove</button>'
        : '<button type="button" class="btn ghost small" data-get="' + escape(entry.id) + '"' +
          (llama.busy ? " disabled" : "") + ">Download</button>";
      return '<div class="model-row' + (entry.installed ? " is-installed" : "") + '">' +
        "<div><strong>" + escape(entry.name) + "</strong><small>" + escape(entry.note) + "</small></div>" +
        '<span class="size">' + entry.size_gb + " GB</span>" + action + "</div>";
    }).join("");

    // --- ollama ----------------------------------------------------------
    var ollama = data.ollama;
    $("ollamaState").textContent = ollama.available ? "reachable" : "not answering";
    $("ollamaState").dataset.s = ollama.available ? "ready" : "missing";
    $("setMuseModel").innerHTML = ollama.models.length
      ? ollama.models.map(function (m) {
          return '<option value="' + escape(m.name) + '">' + escape(m.name) + " · " + m.size_gib + " GiB</option>";
        }).join("")
      : '<option value="">no models installed</option>';
    if (ollama.selected) $("setMuseModel").value = ollama.selected;
    $("museCard").innerHTML = [
      ["Models", String(ollama.models.length), ""],
      ["Loaded now", ollama.loaded.length ? ollama.loaded.join(", ") : "none", ollama.loaded.length ? "" : "ok"]
    ].map(function (row) {
      return "<div><dt>" + row[0] + '</dt><dd class="' + row[2] + '">' + escape(row[1]) + "</dd></div>";
    }).join("");

    // --- the compose-side picker follows whichever backend is active -----
    var active = data.backend;
    $("setWriterBackend").value = data.configured || "auto";
    $("paneLlamacpp").classList.toggle("is-dim", active !== "llamacpp");
    $("paneOllama").classList.toggle("is-dim", active !== "ollama");

    var options, selected;
    if (active === "llamacpp") {
      options = llama.models.map(function (m) { return [m.path, m.file + " · " + m.size_gb + " GiB"]; });
      selected = llama.selected;
    } else {
      options = ollama.models.map(function (m) { return [m.name, m.name + " · " + m.size_gib + " GiB"]; });
      selected = ollama.selected;
    }
    $("museModel").innerHTML = options.length
      ? options.map(function (o) { return '<option value="' + escape(o[0]) + '">' + escape(o[1]) + "</option>"; }).join("")
      : '<option value="">no writer model yet — see Engine</option>';
    if (selected) $("museModel").value = selected;

    var usable = options.length > 0 && (active !== "llamacpp" || llamaReady);
    $("museBtn").disabled = !usable || !!llama.busy;
    $("museTag").textContent = active === "llamacpp" ? "llama.cpp" : "ollama";
    if (!usable) {
      $("museStatus").textContent = active === "llamacpp"
        ? "Install llama.cpp and a writer model under Engine"
        : "Ollama is not reachable — switch the writer to llama.cpp under Engine";
      $("museStatus").classList.add("bad");
    } else if (!llama.busy) {
      $("museStatus").classList.remove("bad");
    }
    if (llama.busy) $("museStatus").textContent = llama.busy;
  }

  function refreshMuse() {
    return api("/api/writer/status").then(function (data) {
      paintWriter(data);
      $("museFree").checked = !!data.free_engine;
      $("setMuseFree").checked = !!data.free_engine;
    }).catch(function () {});
  }

  $("llamaInstall").addEventListener("click", function () {
    api("/api/writer/llamacpp/install", { method: "POST" })
      .then(function () { toast("Downloading the llama.cpp CUDA build (about 400 MB)"); })
      .catch(function (error) { toast(error.message, "bad"); });
  });

  $("llamaCatalog").addEventListener("click", function (event) {
    var get = event.target.closest("[data-get]");
    var drop = event.target.closest("[data-drop]");
    if (get) {
      var body = new FormData();
      body.append("model_id", get.dataset.get);
      api("/api/writer/llamacpp/model", { method: "POST", body: body })
        .then(function () { toast("Downloading the writer model"); })
        .catch(function (error) { toast(error.message, "bad"); });
    } else if (drop) {
      if (!window.confirm("Delete this writer model file?")) return;
      api("/api/writer/llamacpp/model/" + encodeURIComponent(drop.dataset.drop), { method: "DELETE" })
        .then(function () { toast("Writer model deleted"); return refreshMuse(); })
        .catch(function (error) { toast(error.message, "bad"); });
    }
  });

  function addDir(inputId, key, current, refresh) {
    var value = $(inputId).value.trim();
    if (!value) return;
    var dirs = (current || []).concat([value]);
    var payload = {};
    payload[key] = dirs;
    api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify(payload) })
      .then(function () { $(inputId).value = ""; toast("Scanning " + value); return refresh(); })
      .catch(function (error) { toast(error.message, "bad"); });
  }

  $("writerDirAdd").addEventListener("click", function () {
    addDir("writerDir", "writer_dirs", ((STATE.writer || {}).llamacpp || {}).dirs, refreshMuse);
  });
  $("artDirAdd").addEventListener("click", function () {
    addDir("artDir", "art_dirs", (STATE.art || {}).dirs, refreshArt);
  });

  $("setWriterBackend").addEventListener("change", function () {
    api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ writer_backend: this.value })
    }).then(refreshMuse).catch(function (error) { toast(error.message, "bad"); });
  });

  $("setLlamaModel").addEventListener("change", function () {
    api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ llamacpp_model: this.value })
    }).then(refreshMuse).catch(function () {});
  });

  $("museModel").addEventListener("change", function () {
    var active = (STATE.writer || {}).backend;
    var target = active === "llamacpp" ? "llamacpp_model" : "muse_model";
    var payload = {};
    payload[target] = this.value;
    api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify(payload) }).then(refreshMuse).catch(function () {});
  });
  $("museFree").addEventListener("change", function () { $("setMuseFree").checked = this.checked; });

  $("idea").addEventListener("keydown", function (event) {
    if (event.key === "Enter") { event.preventDefault(); writeBrief(); }
  });
  $("museBtn").addEventListener("click", writeBrief);

  function writeBrief() {
    var idea = $("idea").value.trim();
    if (!idea) { $("idea").focus(); return toast("Describe the song in a line first", "bad"); }

    var button = $("museBtn"), started = Date.now();
    button.dataset.busy = "1";
    $("museLabel").textContent = "Writing…";
    $("museStatus").classList.remove("bad");
    $("museStatus").textContent = "loading " + ($("museModel").value || "the writer");
    var ticker = setInterval(function () {
      $("museStatus").textContent = "writing… " + Math.round((Date.now() - started) / 1000) + "s";
    }, 1000);

    api("/api/muse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idea: idea, model: $("museModel").value,
                             free_engine: $("museFree").checked,
                             structure: $("structure").value })
    }).then(function (brief) {
      if (brief.title) $("title").value = brief.title;
      if (brief.style) $("style").value = brief.style;
      if (brief.lyrics) { $("lyrics").value = brief.lyrics; growLyrics(); }
      // The writer sizes this line for whichever cover model is selected, so it
      // travels with the song rather than being rebuilt later.
      STATE.coverPrompt = brief.cover || "";
      $("coverLine").textContent = STATE.coverPrompt ? "Cover: " + STATE.coverPrompt : "";
      var cut = brief.done_reason === "length";
      $("museStatus").textContent = brief.backend + " · " + brief.seconds + "s · " + brief.lines + " lines · " +
        (brief.sections || []).length + " sections" +
        (cut ? " · CUT OFF at the token limit" : "") + " · " +
        (brief.unloaded ? "writer unloaded" : "writer STILL LOADED");
      if (!brief.unloaded || cut) $("museStatus").classList.add("bad");
      toast("Brief written: " + brief.title, "good");
      refreshMuse();
    }).catch(function (error) {
      $("museStatus").textContent = "";
      toast(error.message, "bad");
    }).then(function () {
      clearInterval(ticker);
      button.dataset.busy = "0";
      $("museLabel").textContent = "Write the brief";
    });
  }

  /* ---------------------------------------------------------------- cover */

  function refreshCover() {
    return api("/api/cover/status").then(function (data) {
      $("coverTake").innerHTML = data.takes.length
        ? data.takes.map(function (t) {
            return '<option value="' + escape(t.name) + '">' + escape(t.title) + "</option>";
          }).join("")
        : '<option value="">no take has a score yet</option>';
      if (!data.sheetsage.available) {
        $("coverFromAudio").disabled = true;
        $("coverFile").disabled = true;
        $("coverStatus").textContent =
          "Transcribing a recording needs SheetSage2, which pins different dependencies and " +
          "lives in its own environment (" + data.sheetsage.env_dir + "). See webui/README.md. " +
          "Covering from a take works without it.";
      } else {
        $("coverFromAudio").disabled = false;
        $("coverFile").disabled = false;
        $("coverStatus").textContent = "SheetSage2 ready.";
      }
    }).catch(function () {});
  }

  function applyCoverScore(abc, note) {
    $("abc").value = abc;
    document.querySelector('input[name="cot"][value="melody"]').checked = true;
    $("scoreDrawer").open = true;
    $("coverStatus").textContent = note;
    toast(note, "good");
  }

  $("coverFromTake").addEventListener("click", function () {
    var name = $("coverTake").value;
    if (!name) return toast("No take has a score to cover yet", "bad");
    var body = new FormData();
    body.append("name", name);
    body.append("melody_only", $("coverMelodyOnly").checked ? "true" : "false");
    api("/api/cover/from-take", { method: "POST", body: body }).then(function (data) {
      if (!$("lyrics").value.trim()) { $("lyrics").value = data.lyrics; growLyrics(); }
      applyCoverScore(data.abc, "Melody loaded from “" + data.title + "”. Now write the new style and generate.");
    }).catch(function (error) { toast(error.message, "bad"); });
  });

  $("coverFromAudio").addEventListener("click", function () {
    var file = $("coverFile").files[0];
    if (!file) return toast("Choose an audio file first", "bad");
    var body = new FormData();
    body.append("file", file);
    body.append("melody_only", $("coverMelodyOnly").checked ? "true" : "false");
    $("coverStatus").textContent = "Transcribing " + file.name + " — this runs SheetSage2 on the GPU…";
    $("coverFromAudio").disabled = true;
    api("/api/cover/from-audio", { method: "POST", body: body }).then(function (data) {
      applyCoverScore(data.abc, "Transcribed in " + data.seconds + "s. Check the melody, then write the new style.");
    }).catch(function (error) {
      $("coverStatus").textContent = error.message;
      toast(error.message, "bad");
    }).then(function () { $("coverFromAudio").disabled = false; });
  });

  /* ------------------------------------------------------------------ art */

  function paintArt(data) {
    STATE.art = data;
    var ready = data.installed;
    $("artState").textContent = data.busy ? "working" : (ready ? "installed" : "not installed");
    $("artState").dataset.s = data.busy ? "busy" : (ready ? "ready" : "missing");
    $("artInstall").textContent = ready ? "Reinstall" : "Install";
    $("artInstall").disabled = !!data.busy || !data.supported;
    $("setArtAuto").checked = !!data.auto;

    $("setArtModel").innerHTML = data.models.length
      ? data.models.map(function (m) {
          return '<option value="' + escape(m.path) + '">' + escape(m.file) + " · " + m.size_gb +
                 " GiB · " + escape(m.source) + "</option>";
        }).join("")
      : '<option value="">no checkpoint found yet</option>';
    $("artDirs").textContent = (data.dirs && data.dirs.length)
      ? "Also scanning: " + data.dirs.join("  ·  ")
      : "Only the download folder is scanned. Add a checkpoints folder to reuse models you already have.";
    if (data.selected) $("setArtModel").value = data.selected;

    $("artCatalog").innerHTML = data.catalog.map(function (entry) {
      var action = entry.installed
        ? '<button type="button" class="btn ghost small" data-artdrop="' + escape(entry.file) + '">Remove</button>'
        : '<button type="button" class="btn ghost small" data-artget="' + escape(entry.id) + '"' +
          (data.busy ? " disabled" : "") + ">Download</button>";
      return '<div class="model-row' + (entry.installed ? " is-installed" : "") + '">' +
        "<div><strong>" + escape(entry.name) + "</strong><small>" + escape(entry.note) + "</small></div>" +
        '<span class="size">' + entry.size_gb + " GB</span>" + action + "</div>";
    }).join("");
  }

  function refreshArt() {
    return api("/api/art/status").then(paintArt).catch(function () {});
  }

  $("artInstall").addEventListener("click", function () {
    api("/api/art/install", { method: "POST" })
      .then(function () { toast("Downloading stable-diffusion.cpp (about 900 MB)"); })
      .catch(function (error) { toast(error.message, "bad"); });
  });

  $("artCatalog").addEventListener("click", function (event) {
    var get = event.target.closest("[data-artget]");
    var drop = event.target.closest("[data-artdrop]");
    if (get) {
      var body = new FormData();
      body.append("model_id", get.dataset.artget);
      api("/api/art/model", { method: "POST", body: body })
        .then(function () { toast("Downloading the checkpoint"); })
        .catch(function (error) { toast(error.message, "bad"); });
    } else if (drop) {
      if (!window.confirm("Delete this checkpoint?")) return;
      api("/api/art/model/" + encodeURIComponent(drop.dataset.artdrop), { method: "DELETE" })
        .then(function () { toast("Checkpoint deleted"); return refreshArt(); })
        .catch(function (error) { toast(error.message, "bad"); });
    }
  });

  $("setArtModel").addEventListener("change", function () {
    api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ art_model: this.value }) })
      .then(refreshArt).catch(function () {});
  });

  $("setArtAuto").addEventListener("change", function () {
    api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ art_auto: this.checked }) })
      .then(refreshArt).catch(function () {});
  });

  function drawCover(name, button) {
    if (button) { button.disabled = true; button.textContent = "Drawing…"; }
    api("/api/art/cover/" + encodeURIComponent(name), { method: "POST" })
      .then(function (result) {
        toast("Cover drawn in " + result.seconds + "s", "good");
        return refreshLibrary();
      })
      .catch(function (error) { toast(error.message, "bad"); })
      .then(function () {
        if (button) { button.disabled = false; button.textContent = "Draw cover"; }
        if (STATE.take && STATE.take.name === name) openTake(name);
      });
  }

  /* -------------------------------------------------------------- compose */

  // Verse, Chorus, Bridge and Interlude are the tags YuE2's own score vocabulary
  // documents; the rest are passed through as written, so say so plainly.
  var STRUCTURE_NOTES = {
    simple: "Verse and Chorus only.",
    bridge: "Verse, Chorus and Bridge — the tags YuE2's score vocabulary documents.",
    prechorus: "Adds [Pre-Chorus], which is not one of YuE2's documented tags: it is passed " +
               "through as written and may be sung as an ordinary section.",
    full: "Adds [Intro], [Pre-Chorus] and [Outro]. Only Verse, Chorus, Bridge and Interlude are " +
          "documented tags; the others are passed through as written, and instrumental sections " +
          "carry no lyrics, so the model may or may not leave them wordless."
  };

  function paintStructureHint() {
    $("structureHint").textContent = STRUCTURE_NOTES[$("structure").value] || "";
  }
  $("structure").addEventListener("change", paintStructureHint);
  paintStructureHint();

  // A 10-row box hides the bridge below the fold, which reads as truncated lyrics.
  function growLyrics() {
    var box = $("lyrics");
    box.style.height = "auto";
    box.style.height = Math.min(760, Math.max(180, box.scrollHeight + 4)) + "px";
  }
  $("lyrics").addEventListener("input", growLyrics);

  $("rollSeed").addEventListener("click", function () {
    $("seed").value = Math.floor(Math.random() * 2147483647);
  });

  $("clearForm").addEventListener("click", function () {
    ["title", "style", "lyrics", "abc", "seed", "cfg"].forEach(function (id) { $(id).value = ""; });
    STATE.coverPrompt = "";
    $("coverLine").textContent = "";
    toast("Form cleared");
  });

  $("loadExample").addEventListener("click", function () {
    if (!STATE.examples || !STATE.examples.song) return toast("No example found in examples/song.json", "bad");
    var song = STATE.examples.song;
    $("title").value = "City Lights";
    $("style").value = song.style;
    $("lyrics").value = song.lyrics;
    $("seed").value = song.seed;
    document.querySelector('input[name="cot"][value="' + (song.cot || "full") + '"]').checked = true;
    toast("Example song loaded");
  });

  Array.prototype.forEach.call(document.querySelectorAll("[data-abc]"), function (btn) {
    btn.addEventListener("click", function () {
      var key = btn.dataset.abc;
      if (!key) { $("abc").value = ""; return toast("Score removed"); }
      if (!STATE.examples || !STATE.examples[key]) return toast("That example is not in examples/", "bad");
      $("abc").value = STATE.examples[key];
      toast("Score loaded — melody mode suits covers best");
    });
  });

  $("composeForm").addEventListener("submit", function (event) {
    event.preventDefault();
    var cot = document.querySelector('input[name="cot"]:checked').value;
    var abc = $("abc").value.trim();
    if (abc && cot === "off") {
      return toast("Direct mode ignores a supplied score — pick full or melody", "bad");
    }
    var title = $("title").value.trim();
    var slug = (title || "song").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "song";
    var seedRaw = $("seed").value.trim();
    var cfgRaw = $("cfg").value.trim();

    var body = {
      style: $("style").value.trim(),
      lyrics: $("lyrics").value,
      cot: cot,
      seed: seedRaw === "" ? null : parseInt(seedRaw, 10),
      id: slug,
      title: title,
      abc: abc || null,
      cfg_scale: cfgRaw === "" ? null : parseFloat(cfgRaw),
      cover: STATE.coverPrompt || "",
      abc_sampling: samplingOverrides("abc"),
      semantic_sampling: samplingOverrides("semantic")
    };

    $("generateBtn").disabled = true;
    api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (data) {
      watchJob(data.job);
      show("take");
      toast("Queued: " + data.job.title);
    }).catch(function (error) {
      toast(error.message, "bad");
    }).then(function () {
      // Runs queue server-side, so the button only stays down for the request
      // itself. It must never depend on a later event to come back to life.
      $("generateBtn").disabled = false;
    });
  });

  /* ------------------------------------------------------------ run chain */

  var runTimer = null;

  function watchJob(job) {
    STATE.job = job;
    STATE.take = null;
    STATE.peaks = null;
    STATE.abcRendered = "";
    $("takeBody").classList.add("is-hidden");
    $("takeEmpty").classList.add("is-hidden");
    $("chain").hidden = false;
    $("takeActions").innerHTML = "";
    paintJob(job);
    if (!runTimer) runTimer = setInterval(tickClock, 250);
  }

  function tickClock() {
    var job = STATE.job;
    if (!job) return;
    if (job.state === "running" && job.started) {
      $("runClock").textContent = clock((Date.now() / 1000) - job.started);
      var setupClock = $("setupClock");
      if (setupClock && job.setup_at) setupClock.textContent = clock((Date.now() / 1000) - job.setup_at);
    } else if (job.started && job.finished) {
      $("runClock").textContent = clock(job.finished - job.started);
    }
    if (job.state !== "running" && job.state !== "queued") {
      clearInterval(runTimer);
      runTimer = null;
    }
  }

  function paintJob(job) {
    STATE.job = job;
    var stateLabels = { queued: "Queued", running: "Running", done: "Complete", failed: "Failed", cancelled: "Cancelled" };
    $("runState").textContent = stateLabels[job.state] || job.state;
    $("runState").dataset.s = job.state;
    $("cancelRun").classList.toggle("is-hidden", job.state !== "running" && job.state !== "queued");
    $("takeEyebrow").textContent = job.setup ? job.setup : (stateLabels[job.state] || job.state);
    $("takeTitle").textContent = job.title;

    $("runError").classList.toggle("is-hidden", !job.error);
    $("runError").textContent = job.error || "";

    var finished = job.state !== "running" && job.state !== "queued";
    var html = "";

    // Setup work — resolving files, loading weights, loading the decoder — is a
    // real step the run waits on, so it gets a row of its own while it lasts.
    if (job.setup) {
      html += '<li class="stage" data-s="running"><span class="stage-num">·</span>' +
        '<div class="stage-main"><div class="stage-label">' + escape(job.setup) + "</div>" +
        '<div class="stage-note">Preparing the engine</div>' +
        '<div class="meter indeterminate"><i></i></div></div>' +
        '<div class="stage-read" id="setupClock">' +
        (job.setup_at ? clock((Date.now() / 1000) - job.setup_at) : "") + "</div></li>";
    }

    job.stages.forEach(function (stage, index) {
      var shown = (finished && stage.state === "waiting") ? "skipped" : stage.state;
      var read = "";
      if (stage.completed || stage.total) {
        var rate = stage.seconds > 0.4 ? (stage.completed / stage.seconds) : 0;
        read = "<b>" + stage.completed.toLocaleString() + "</b>";
        if (stage.total) read += " / " + stage.total.toLocaleString();
        if (stage.unit) read += " " + stage.unit;
        if (stage.state === "running" && stage.total && stage.completed > 0) {
          // A long song's synthesis runs for minutes at full tilt; without a
          // countdown that is indistinguishable from a hang.
          var left = (stage.total - stage.completed) * (stage.seconds / stage.completed);
          read += "<br>" + clock(left) + " left";
        } else if (rate > 0 && stage.state === "running") {
          read += "<br>" + rate.toFixed(1) + "/s";
        } else if (stage.seconds > 0) {
          read += "<br>" + stage.seconds.toFixed(1) + "s";
        }
      }
      var pct = stage.total ? Math.min(100, (stage.completed / stage.total) * 100) : 0;
      var meter = "";
      if (stage.state === "running") {
        meter = stage.total
          ? '<div class="meter"><i style="width:' + pct.toFixed(1) + '%"></i></div>'
          : '<div class="meter indeterminate"><i></i></div>';
      } else if (stage.state === "completed") {
        meter = '<div class="meter"><i style="width:100%;background:var(--patina-dim)"></i></div>';
      }
      var note = shown === "skipped" ? "Skipped in this mode" : stage.note;
      html += '<li class="stage" data-s="' + shown + '">' +
        '<span class="stage-num">' + (index + 1) + "</span>" +
        '<div class="stage-main"><div class="stage-label">' + escape(stage.label) + "</div>" +
        '<div class="stage-note">' + escape(note) + "</div>" + meter + "</div>" +
        '<div class="stage-read">' + read + "</div></li>";
    });
    $("stages").innerHTML = html;

    if (job.abc_partial && !STATE.take) {
      $("takeBody").classList.remove("is-hidden");
      $("scorePanel").classList.remove("is-hidden");
      $("metaGrid").classList.add("is-hidden");
      $("scoreStaff").classList.add("is-hidden");
      $("scoreAbc").classList.remove("is-hidden");
      $("scoreAbc").textContent = job.abc_partial;
      $("scoreAbc").classList.toggle("paper-live", job.state === "running");
    }

    if (job.state === "failed" && job.error) toastOnce(job.id, job.error, "bad");
    if (job.state === "done" && job.take) {
      toastOnce(job.id, "Song complete — " + job.title, "good");
      refreshLibrary().then(function () { openTake(job.take); });
    }
  }

  var toasted = {};
  function toastOnce(key, message, kind) {
    if (toasted[key]) return;
    toasted[key] = true;
    toast(message, kind);
  }

  $("cancelRun").addEventListener("click", function () {
    if (!STATE.job) return;
    api("/api/jobs/" + STATE.job.id + "/cancel", { method: "POST" })
      .then(function () { toast("Cancelling after the current step"); })
      .catch(function (error) { toast(error.message, "bad"); });
  });

  /* ---------------------------------------------------------------- take */

  function openTake(name) {
    var take = STATE.takes.filter(function (t) { return t.name === name; })[0];
    if (!take) return;
    STATE.take = take;
    STATE.peaks = null;
    STATE.job = null;
    $("chain").hidden = true;
    $("takeEmpty").classList.add("is-hidden");
    $("takeBody").classList.remove("is-hidden");
    $("metaGrid").classList.remove("is-hidden");
    $("scorePanel").classList.toggle("is-hidden", !take.score);
    $("scoreAbc").classList.remove("paper-live");
    $("metaCover").textContent = take.cover_prompt ? "Cover: " + take.cover_prompt : "";

    $("takeEyebrow").textContent = ago(take.created) + " · seed " + take.seed;
    $("takeTitle").textContent = take.title;
    $("takeActions").innerHTML = '<button type="button" class="btn ghost" id="drawCover">' +
      (take.has_cover ? "Redraw cover" : "Draw cover") + "</button>" +
      '<a class="btn ghost" href="/api/library/' + encodeURIComponent(take.name) +
      '/file/result.json" download>Run details</a>' +
      (take.score ? '<a class="btn ghost" href="/api/library/' + encodeURIComponent(take.name) +
        '/file/score.abc" download>Score .abc</a>' : "");
    $("drawCover").addEventListener("click", function () { drawCover(take.name, this); });

    $("coverWrap").classList.toggle("is-hidden", !take.has_cover);
    if (take.has_cover) {
      $("coverImg").src = "/api/library/" + encodeURIComponent(take.name) + "/cover?t=" + Date.now();
    }

    var audio = $("audio");
    audio.src = "/api/library/" + encodeURIComponent(take.name) + "/audio";
    $("playbar").classList.remove("is-empty");
    $("playbarTitle").textContent = take.title;
    $("playbarStyle").textContent = take.style;
    var thumb = $("playbarCover");
    thumb.hidden = !take.has_cover;
    if (take.has_cover) thumb.src = "/api/library/" + encodeURIComponent(take.name) + "/cover";
    $("dlAudio").href = audio.src;
    $("dlAudio").setAttribute("download", take.name + ".flac");
    $("timeTotal").textContent = clock(take.seconds);
    $("timeNow").textContent = "0:00";
    $("playGlyph").textContent = "▶";

    var modes = { full: "Full plan", melody: "Melody only", off: "Direct" };
    var cells = [
      ["Length", clock(take.seconds)],
      ["Mode", modes[take.cot] || take.cot],
      ["Seed", take.seed],
      ["Decoder", take.vae === "legacy" ? "legacy" : "standard"],
      ["Render time", take.e2e_seconds ? clock(take.e2e_seconds) : "—"],
      ["Sample rate", (take.sample_rate / 1000) + " kHz"]
    ];
    var truncated = take.truncated && (take.truncated.abc || take.truncated.semantic);
    if (truncated) cells.push(["Limit", "hit token cap"]);
    if (take.provided_score) cells.push(["Score", "supplied"]);

    $("metaGrid").innerHTML = cells.map(function (cell) {
      var warn = cell[0] === "Limit" ? ' class="warn"' : "";
      return "<div><dt>" + cell[0] + "</dt><dd" + warn + ">" + escape(String(cell[1])) + "</dd></div>";
    }).join("");

    $("metaStyle").textContent = take.style;
    // Section tags carry the structure, so they are marked rather than escaped flat.
    $("metaLyrics").innerHTML = (take.lyrics || "").split(/\r?\n/).map(function (line) {
      return line.trim().charAt(0) === "[" ? "<b>" + escape(line) + "</b>" : escape(line);
    }).join("\n");

    renderScore(take.score);
    loadPeaks(audio.src);
    paintLibrary();
    show("take");
  }

  function renderScore(abc) {
    $("scoreAbc").textContent = abc || "";
    if (!abc) return;
    if (window.__abcjsFailed || typeof window.ABCJS === "undefined") {
      // No renderer available: the ABC text is the score.
      $("scoreStaff").classList.add("is-hidden");
      $("scoreAbc").classList.remove("is-hidden");
      document.querySelector('.stab[data-score="staff"]').disabled = true;
      return;
    }
    try {
      window.ABCJS.renderAbc("scoreStaff", abc, {
        responsive: "resize",
        staffwidth: 700,
        paddingtop: 4,
        paddingbottom: 10,
        foregroundColor: "#2a2318"
      });
      STATE.abcRendered = abc;
    } catch (error) {
      $("scoreStaff").textContent = "This score could not be engraved; read it as ABC.";
    }
  }

  Array.prototype.forEach.call(document.querySelectorAll(".stab"), function (tab) {
    tab.addEventListener("click", function () {
      Array.prototype.forEach.call(document.querySelectorAll(".stab"), function (t) {
        t.classList.toggle("is-active", t === tab);
      });
      var staff = tab.dataset.score === "staff";
      $("scoreStaff").classList.toggle("is-hidden", !staff);
      $("scoreAbc").classList.toggle("is-hidden", staff);
    });
  });

  $("copyLyrics").addEventListener("click", function () {
    var text = STATE.take ? STATE.take.lyrics : "";
    if (!text) return;
    navigator.clipboard.writeText(text)
      .then(function () { toast("Lyrics copied"); })
      .catch(function () { toast("The browser refused clipboard access", "bad"); });
  });

  $("reuseScore").addEventListener("click", function () {
    var take = STATE.take;
    if (!take || !take.score) return;
    $("abc").value = take.score;
    $("style").value = take.style;
    $("lyrics").value = take.lyrics;
    $("title").value = take.title + " (edit)";
    $("scoreDrawer").open = true;
    show("compose");
    $("abc").focus();
    toast("Score copied into the form — edit it, then generate");
  });

  /* ------------------------------------------------------- cover visualiser */
  /* A ring of bars around the sleeve, driven by the real spectrum. The audio
     graph is built once: a MediaElementSource can only be created per element,
     and it must reach the destination or the sound disappears. */

  var VIZ = { analyser: null, data: null, frame: 0 };

  function ensureAnalyser() {
    if (VIZ.analyser || !window.AudioContext) return VIZ.analyser;
    try {
      var context = new AudioContext();
      var source = context.createMediaElementSource(audio);
      var analyser = context.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.75;
      source.connect(analyser);
      analyser.connect(context.destination);
      VIZ.analyser = analyser;
      VIZ.context = context;
      VIZ.data = new Uint8Array(analyser.frequencyBinCount);
    } catch (error) {
      VIZ.analyser = null;   // visuals are a nicety; playback still works
    }
    return VIZ.analyser;
  }

  function drawViz() {
    var canvas = $("viz");
    var analyser = VIZ.analyser;
    if (!analyser || audio.paused) { VIZ.frame = 0; return; }

    var dpr = window.devicePixelRatio || 1;
    var size = canvas.clientWidth;
    if (!size) { VIZ.frame = requestAnimationFrame(drawViz); return; }
    if (canvas.width !== Math.floor(size * dpr)) {
      canvas.width = canvas.height = Math.floor(size * dpr);
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    analyser.getByteFrequencyData(VIZ.data);
    var bins = VIZ.data;
    var bars = 72;
    var centre = size / 2;
    var radius = size * 0.40;
    var maxBar = size * 0.085;

    ctx.lineCap = "round";
    for (var i = 0; i < bars; i++) {
      // Low bins hold most of the energy, so read them on a curve.
      var index = Math.floor(Math.pow(i / bars, 1.7) * (bins.length - 1));
      var level = bins[index] / 255;
      var length = 2 + level * maxBar;
      var angle = (i / bars) * Math.PI * 2 - Math.PI / 2;
      var x1 = centre + Math.cos(angle) * radius;
      var y1 = centre + Math.sin(angle) * radius;
      var x2 = centre + Math.cos(angle) * (radius + length);
      var y2 = centre + Math.sin(angle) * (radius + length);
      ctx.strokeStyle = "rgba(232, 163, 61, " + (0.35 + level * 0.65) + ")";
      ctx.lineWidth = size * 0.008;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // A progress arc on the same circle, so position is readable at a glance.
    if (isFinite(audio.duration) && audio.duration > 0) {
      ctx.strokeStyle = "rgba(240, 231, 216, .85)";
      ctx.lineWidth = size * 0.006;
      ctx.beginPath();
      ctx.arc(centre, centre, radius - size * 0.02, -Math.PI / 2,
              -Math.PI / 2 + (audio.currentTime / audio.duration) * Math.PI * 2);
      ctx.stroke();
    }

    VIZ.frame = requestAnimationFrame(drawViz);
  }

  function startViz() {
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (ensureAnalyser() && VIZ.context && VIZ.context.state === "suspended") VIZ.context.resume();
    $("coverWrap").classList.add("is-playing");
    if (!VIZ.frame) drawViz();
  }

  function stopViz() {
    $("coverWrap").classList.remove("is-playing");
    if (VIZ.frame) { cancelAnimationFrame(VIZ.frame); VIZ.frame = 0; }
  }

  /* -------------------------------------------------------------- player */

  var audio = $("audio");

  function togglePlay() {
    if (audio.paused) audio.play().catch(function () { toast("Your browser would not play this FLAC", "bad"); });
    else audio.pause();
  }

  $("playBtn").addEventListener("click", togglePlay);

  // Volume survives reloads; a browser remembering nothing is a small annoyance
  // people notice every single time.
  var storedVolume = null;
  try { storedVolume = localStorage.getItem("yue2.volume"); } catch (error) { storedVolume = null; }
  audio.volume = storedVolume === null ? 1 : Math.min(1, Math.max(0, parseFloat(storedVolume)));
  $("volume").value = audio.volume;

  function paintVolume() {
    var level = audio.muted ? 0 : audio.volume;
    $("muteGlyph").textContent = level === 0 ? "🔇" : (level < 0.5 ? "🔉" : "🔊");
    $("volume").value = level;
  }

  $("volume").addEventListener("input", function () {
    audio.volume = parseFloat(this.value);
    audio.muted = audio.volume === 0;
    try { localStorage.setItem("yue2.volume", String(audio.volume)); } catch (error) { /* private mode */ }
    paintVolume();
  });

  $("muteBtn").addEventListener("click", function () {
    audio.muted = !audio.muted;
    if (!audio.muted && audio.volume === 0) audio.volume = 0.7;
    paintVolume();
  });

  paintVolume();
  audio.addEventListener("play", function () { $("playGlyph").textContent = "❚❚"; startViz(); });
  audio.addEventListener("pause", function () { $("playGlyph").textContent = "▶"; stopViz(); });
  audio.addEventListener("ended", function () { $("playGlyph").textContent = "▶"; stopViz(); });
  audio.addEventListener("timeupdate", function () {
    $("timeNow").textContent = clock(audio.currentTime);
    drawWave();
  });
  audio.addEventListener("loadedmetadata", function () {
    if (isFinite(audio.duration)) $("timeTotal").textContent = clock(audio.duration);
  });

  $("wave").addEventListener("click", function (event) {
    var rect = this.getBoundingClientRect();
    var ratio = (event.clientX - rect.left) / rect.width;
    if (isFinite(audio.duration)) audio.currentTime = ratio * audio.duration;
  });

  function loadPeaks(url) {
    STATE.peaks = null;
    drawWave();
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    fetch(url).then(function (r) { return r.arrayBuffer(); }).then(function (buffer) {
      var context = new Ctx();
      return context.decodeAudioData(buffer).then(function (decoded) {
        var channel = decoded.getChannelData(0);
        var buckets = 900, size = Math.floor(channel.length / buckets), peaks = new Float32Array(buckets);
        for (var i = 0; i < buckets; i++) {
          var max = 0;
          for (var j = 0; j < size; j += 3) {
            var value = Math.abs(channel[i * size + j]);
            if (value > max) max = value;
          }
          peaks[i] = max;
        }
        STATE.peaks = peaks;
        context.close();
        drawWave();
      });
    }).catch(function () { /* waveform is a nicety; the transport still works */ });
  }

  function drawWave() {
    var canvas = $("wave");
    if (!canvas.clientWidth) return;
    var dpr = window.devicePixelRatio || 1;
    var width = canvas.clientWidth, height = 56;
    if (canvas.width !== Math.floor(width * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    var middle = height / 2;
    ctx.strokeStyle = "rgba(240,231,216,.08)";
    ctx.beginPath();
    ctx.moveTo(0, middle + 0.5);
    ctx.lineTo(width, middle + 0.5);
    ctx.stroke();

    var progress = (isFinite(audio.duration) && audio.duration > 0) ? audio.currentTime / audio.duration : 0;

    if (!STATE.peaks) {
      ctx.fillStyle = "rgba(240,231,216,.22)";
      ctx.font = '11px "IBM Plex Mono", monospace';
      ctx.fillText("reading waveform…", 12, middle + 4);
      return;
    }

    var peaks = STATE.peaks, bars = Math.min(peaks.length, Math.floor(width / 3));
    var step = peaks.length / bars;
    for (var i = 0; i < bars; i++) {
      var value = peaks[Math.floor(i * step)];
      var barHeight = Math.max(1.5, value * (height - 16));
      var x = i * (width / bars);
      ctx.fillStyle = (i / bars) <= progress ? "#e8a33d" : "rgba(240,231,216,.24)";
      ctx.fillRect(x, middle - barHeight / 2, Math.max(1, width / bars - 1), barHeight);
    }

    if (progress > 0) {
      ctx.fillStyle = "#f0e7d8";
      ctx.fillRect(progress * width - 0.5, 6, 1.5, height - 12);
    }
  }

  window.addEventListener("resize", drawWave);

  /* -------------------------------------------------------------- library */

  function refreshLibrary() {
    return api("/api/library").then(function (data) {
      STATE.takes = data.takes;
      paintLibrary();
    });
  }

  function paintLibrary() {
    var list = $("libList");
    $("libCount").textContent = STATE.takes.length;
    if (!STATE.takes.length) {
      list.innerHTML = '<div class="lib-empty">No takes yet.<br>Every finished song lands here.</div>';
      return;
    }
    list.innerHTML = STATE.takes.map(function (take) {
      var active = STATE.take && STATE.take.name === take.name ? " is-active" : "";
      var thumb = take.has_cover
        ? '<img class="take-thumb" src="/api/library/' + encodeURIComponent(take.name) + '/cover" alt="" />'
        : "";
      return '<article class="take' + active + '" data-name="' + escape(take.name) + '" tabindex="0">' +
        '<button class="take-del" data-del="' + escape(take.name) + '" title="Delete take" aria-label="Delete take">✕</button>' +
        '<div class="take-row">' + thumb + "<div>" +
        '<div class="take-title">' + escape(take.title) + "</div>" +
        '<div class="take-style">' + escape(take.style) + "</div></div></div>" +
        '<div class="take-foot"><span class="tag ' + take.cot + '">' + take.cot + "</span>" +
        "<span>" + clock(take.seconds) + "</span><span>" + ago(take.created) + "</span></div></article>";
    }).join("");
  }

  $("libList").addEventListener("click", function (event) {
    var del = event.target.closest("[data-del]");
    if (del) {
      event.stopPropagation();
      var name = del.dataset.del;
      if (!window.confirm("Delete this take and its artifacts?")) return;
      api("/api/library/" + encodeURIComponent(name), { method: "DELETE" }).then(function () {
        if (STATE.take && STATE.take.name === name) {
          STATE.take = null;
          $("takeBody").classList.add("is-hidden");
          $("takeEmpty").classList.remove("is-hidden");
          $("takeTitle").textContent = "Nothing playing";
          $("takeEyebrow").textContent = "No run yet";
          $("takeActions").innerHTML = "";
        }
        return refreshLibrary();
      }).then(function () { toast("Take deleted"); })
        .catch(function (error) { toast(error.message, "bad"); });
      return;
    }
    var card = event.target.closest(".take");
    if (card) openTake(card.dataset.name);
  });

  $("libList").addEventListener("keydown", function (event) {
    var card = event.target.closest(".take");
    if (card && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      openTake(card.dataset.name);
    }
  });

  /* --------------------------------------------------------------- engine */

  var SETTING_FIELDS = [
    ["setModel", "model", "text"], ["setVae", "vae", "text"], ["setDevice", "device", "text"],
    ["setBackend", "backend", "text"], ["setQuant", "quantization", "text"],
    ["setBudget", "memory_budget_gib", "number"], ["setOde", "ode_steps", "int"],
    ["setOffload", "offload_ar", "bool"], ["setOffline", "offline", "bool"],
    ["setOllama", "ollama_url", "text"], ["setMuseModel", "muse_model", "text"],
    ["setMuseFree", "muse_free_engine", "bool"]
  ];

  function paintSettings(settings) {
    STATE.settings = settings;
    SETTING_FIELDS.forEach(function (field) {
      var el = $(field[0]);
      if (field[2] === "bool") el.checked = !!settings[field[1]];
      else el.value = settings[field[1]];
    });
  }

  $("saveSettings").addEventListener("click", function () {
    var body = {};
    SETTING_FIELDS.forEach(function (field) {
      var el = $(field[0]);
      if (field[2] === "bool") body[field[1]] = el.checked;
      else if (field[2] === "number") body[field[1]] = parseFloat(el.value);
      else if (field[2] === "int") body[field[1]] = parseInt(el.value, 10);
      else body[field[1]] = el.value.trim();
    });
    api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (data) {
      paintSettings(data.settings);
      $("museModel").value = data.settings.muse_model || $("museModel").value;
      $("museFree").checked = !!data.settings.muse_free_engine;
      toast(data.reloaded ? "Saved — the song model reloads on the next run"
                          : (data.saved ? "Saved" : "No changes to save"));
    }).catch(function (error) { toast(error.message, "bad"); });
  });

  $("freeVram").addEventListener("click", function () {
    api("/api/free", { method: "POST" }).then(function (data) {
      var before = data.before ? data.before.used_gib : "?";
      var after = data.after ? data.after.used_gib : "?";
      toast("VRAM " + before + " GiB -> " + after + " GiB", "good");
      paintVram(data.after);
    }).catch(function (error) { toast(error.message, "bad"); });
  });

  $("preloadBtn").addEventListener("click", function () {
    api("/api/load", { method: "POST" })
      .then(function () { toast("Loading the model — first run downloads weights"); })
      .catch(function (error) { toast(error.message, "bad"); });
  });

  /* ---------------------------------------------------------------- boot */

  function resync() {
    return api("/api/state").then(function (data) {
      paintEngine(data.engine);
      paintHardware(data.hardware);
      paintVram(data.vram);
      var live = data.jobs.filter(function (j) { return j.state === "running" || j.state === "queued"; })[0];
      if (live) paintJob(live);
    }).catch(function () {});
  }

  function connect() {
    var source = new EventSource("/api/events");
    source.addEventListener("engine", function (event) { paintEngine(JSON.parse(event.data)); });
    source.addEventListener("job", function (event) {
      var job = JSON.parse(event.data);
      if (!STATE.job || STATE.job.id === job.id) paintJob(job);
    });
    source.addEventListener("library", function () { refreshLibrary(); refreshCover(); });
    source.addEventListener("art", function (event) {
      var data = JSON.parse(event.data);
      if (data.error) toast(data.error, "bad");
      if (data.busy) $("artState").textContent = data.busy;
      if (!data.busy) refreshArt();
    });
    source.addEventListener("writer", function (event) {
      var data = JSON.parse(event.data);
      if (data.error) { toast(data.error, "bad"); }
      if (data.busy) { $("museStatus").textContent = data.busy; }
      if (!data.busy) { refreshMuse(); }
    });
    // A restarted server or a dropped stream means missed events, so re-read the
    // real state on every (re)connection rather than trusting stale UI.
    source.onopen = function () { resync(); };
    source.onerror = function () { /* EventSource retries on its own */ };
    // Belt and braces: if the stream is dead, keep the page honest anyway.
    setInterval(function () {
      if (source.readyState === 2) resync();
    }, 10000);
  }

  window.addEventListener("error", function (event) {
    toast("Page error: " + (event.message || "unknown") + " — reload if things stop responding", "bad");
  });

  api("/api/state").then(function (data) {
    STATE.defaults = data.defaults;
    buildKnobs();
    paintSettings(data.settings);
    paintEngine(data.engine);
    paintHardware(data.hardware);
    $("outputPath").textContent = "Takes are written to " + data.outputs;
    if (!data.hardware.cuda) {
      toast("No CUDA device visible — generation needs an NVIDIA GPU with BF16", "bad");
    }
    var live = data.jobs.filter(function (j) { return j.state === "running" || j.state === "queued"; })[0];
    if (live) { watchJob(live); show("take"); }
    connect();
  }).catch(function (error) { toast("Could not reach the server: " + error.message, "bad"); });

  api("/api/examples").then(function (data) { STATE.examples = data; }).catch(function () {});
  refreshLibrary().catch(function () {});
  refreshMuse();
  refreshCover();
  refreshArt();

  // Keep the VRAM readout honest while runs come and go.
  setInterval(function () {
    api("/api/vram").then(function (data) { paintVram(data.vram); }).catch(function () {});
  }, 5000);
})();

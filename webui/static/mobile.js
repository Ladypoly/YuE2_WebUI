/* YuE2 on a phone. Two screens, one player, and the decisions already made:
   the full plan, the local writer, no sampling knobs. Everything here talks to
   the same API the console does. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var audio = document.getElementById("audio");
  var STATE = { mode: "simple", job: null, takes: [], playing: null, viewing: null,
                saved: [], vocabulary: null };

  // Section tags YuE2 reads, and the plan it always makes here.
  var STRUCTURE = "bridge";
  var COT = "full";

  function api(path, options) {
    return fetch(path, options).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          if (r.status === 401 && body.pin_required) openPin(body.detail);
          throw new Error(body.detail || (r.status + " " + r.statusText));
        });
      }
      return r.status === 204 ? null : r.json();
    });
  }

  function post(path, payload) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(payload) });
  }

  function escape(text) {
    return String(text === undefined || text === null ? "" : text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  var toastTimer;
  function toast(message, kind) {
    var box = $("toast");
    box.textContent = message;
    box.className = "toast is-on" + (kind === "bad" ? " bad" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { box.className = "toast"; }, kind === "bad" ? 6000 : 3200);
  }

  function clock(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    var m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  /* ---------------------------------------------------------------- pin */

  function openPin(detail) {
    var gate = $("pinGate");
    if (!gate.hidden) return;
    gate.hidden = false;
    if (detail) $("pinHint").textContent = detail;
    $("pinInput").focus();
  }

  $("pinForm").addEventListener("submit", function (event) {
    event.preventDefault();
    var pin = $("pinInput").value.trim();
    if (!pin) return;
    $("pinHint").textContent = "checking…";
    fetch("/api/unlock", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ pin: pin }) })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (b) { throw new Error(b.detail || "Wrong PIN"); });
        window.location.reload();
      })
      .catch(function (error) { $("pinHint").textContent = error.message; });
  });

  /* --------------------------------------------------------------- tabs */

  function show(which) {
    ["create", "listen", "song"].forEach(function (name) {
      $("screen-" + name).classList.toggle("is-on", which === name);
    });
    // The song view belongs to the listen tab, so that tab stays lit.
    $("tabCreate").setAttribute("aria-selected", which === "create" ? "true" : "false");
    $("tabListen").setAttribute("aria-selected", which !== "create" ? "true" : "false");
    // The shell does not scroll; the middle does.
    $("main").scrollTop = 0;
  }
  $("tabCreate").addEventListener("click", function () { show("create"); });
  $("tabListen").addEventListener("click", function () { show("listen"); refreshLibrary(); });

  /* ------------------------------------------------------- simple/custom */

  function setMode(mode) {
    STATE.mode = mode;
    $("modeSimple").setAttribute("aria-pressed", mode === "simple" ? "true" : "false");
    $("modeCustom").setAttribute("aria-pressed", mode === "custom" ? "true" : "false");
    $("paneSimple").hidden = mode !== "simple";
    $("paneCustom").hidden = mode !== "custom";
    $("sheetToStyle").hidden = mode !== "custom";
    $("go").textContent = mode === "simple" ? "Write and make the song" : "Make the song";
    $("sheetNote").textContent = mode === "simple"
      ? "Optional. Whatever you set here the writer has to honour."
      : "Optional. Add the musical half to your style prompt with the button below.";
    try { localStorage.setItem("yue2m.mode", mode); } catch (error) {}
  }
  $("modeSimple").addEventListener("click", function () { setMode("simple"); });
  $("modeCustom").addEventListener("click", function () { setMode("custom"); });

  /* --------------------------------------------------------- song sheet */

  // Nine fields is a lot on a phone, so they arrive in three named drawers.
  var GROUPS = [
    { title: "Sound", fields: ["genre", "tempo", "meter", "key"] },
    { title: "Voice", fields: ["voice", "language", "lyrics"] },
    { title: "Words and length", fields: ["theme", "length"] }
  ];

  function sheetValues() {
    var chosen = {};
    Array.prototype.forEach.call(document.querySelectorAll("[data-sheet]"), function (select) {
      if (select.value) chosen[select.dataset.sheet] = select.value;
    });
    return chosen;
  }

  function paintSheetCounts() {
    var chosen = sheetValues();
    GROUPS.forEach(function (group, index) {
      var count = group.fields.filter(function (name) { return chosen[name]; }).length;
      var label = $("sheetCount" + index);
      if (!label) return;
      label.textContent = count ? count + " set" : "";
      label.dataset.on = count ? "1" : "0";
    });
    try { localStorage.setItem("yue2m.sheet", JSON.stringify(chosen)); } catch (error) {}
  }

  function buildSheet(vocabulary) {
    STATE.vocabulary = vocabulary;
    var saved = {};
    try { saved = JSON.parse(localStorage.getItem("yue2m.sheet") || "{}"); } catch (error) {}

    $("sheetGroups").innerHTML = GROUPS.map(function (group, index) {
      var body = group.fields.map(function (name) {
        var field = vocabulary.fields[name];
        if (!field) return "";
        var options = ['<option value="">any</option>'].concat(field.options.map(function (value) {
          return '<option value="' + escape(value) + '">' + escape(value) + "</option>";
        })).join("");
        return '<label class="f"><span>' + escape(field.label) + "</span>" +
               '<select data-sheet="' + name + '">' + options + "</select></label>";
      }).join("");
      return '<details class="sheet-group"><summary>' + escape(group.title) +
        '<span class="count" id="sheetCount' + index + '" data-on="0"></span></summary>' +
        '<div class="sheet-body">' + body + "</div></details>";
    }).join("");

    Array.prototype.forEach.call(document.querySelectorAll("[data-sheet]"), function (select) {
      var name = select.dataset.sheet;
      if (saved[name] && vocabulary.fields[name].options.indexOf(saved[name]) >= 0) {
        select.value = saved[name];
      }
      select.addEventListener("change", paintSheetCounts);
    });
    paintSheetCounts();
  }

  $("sheetClear").addEventListener("click", function () {
    Array.prototype.forEach.call(document.querySelectorAll("[data-sheet]"), function (select) {
      select.value = "";
    });
    paintSheetCounts();
    toast("Sheet cleared");
  });

  $("sheetToStyle").addEventListener("click", function () {
    post("/api/sheet/style", { sheet: sheetValues() }).then(function (data) {
      if (!data.style) return toast("Nothing musical picked yet", "bad");
      var current = $("style").value.trim();
      $("style").value = current ? data.style + ", " + current : data.style;
      toast("Added to the style");
    }).catch(function (error) { toast(error.message, "bad"); });
  });

  /* ------------------------------------------------------------- the run */

  var STEP_LABELS = { plan: "Planning the score", semantic: "Writing the music",
                      synthesize: "Making the sound", decode: "Rendering audio" };
  var runStarted = 0, runTicker = null;

  function paintJob(job) {
    STATE.job = job;
    var live = job.state === "running" || job.state === "queued";
    $("run").hidden = !live && job.state !== "failed";
    $("go").disabled = live;
    if (job.state === "failed") {
      $("runTitle").textContent = "That run failed";
      $("runSteps").innerHTML = '<li class="step" data-s="failed">' + escape(job.error || "Unknown error") + "</li>";
      $("runBar").style.width = "0%";
      $("cancel").hidden = true;
      stopClock();
      return;
    }
    if (!live) {
      stopClock();
      return;
    }
    $("cancel").hidden = false;
    $("runTitle").textContent = job.setup || job.title || "Working";
    var done = 0;
    $("runSteps").innerHTML = (job.stages || []).map(function (stage) {
      if (stage.state === "completed") done += 1;
      var read = "";
      if (stage.state === "running" && stage.total) {
        read = Math.round((stage.completed / stage.total) * 100) + "%";
      } else if (stage.state === "running" && stage.completed) {
        read = stage.completed.toLocaleString();
      }
      return '<li class="step" data-s="' + escape(stage.state) + '"><span class="dot"></span>' +
        escape(STEP_LABELS[stage.key] || stage.label) + '<span class="num">' + read + "</span></li>";
    }).join("");
    $("runBar").style.width = Math.round((done / 4) * 100) + "%";
    if (!runTicker) {
      runStarted = job.started ? job.started * 1000 : Date.now();
      runTicker = setInterval(function () {
        $("runClock").textContent = clock((Date.now() - runStarted) / 1000);
      }, 1000);
    }
  }

  function stopClock() { clearInterval(runTicker); runTicker = null; }

  $("cancel").addEventListener("click", function () {
    if (!STATE.job) return;
    api("/api/jobs/" + encodeURIComponent(STATE.job.id) + "/cancel", { method: "POST" })
      .then(function () { toast("Stopping"); })
      .catch(function (error) { toast(error.message, "bad"); });
  });

  /* ------------------------------------------------------------ making it */

  function generate(spec) {
    return post("/api/generate", spec).then(function (data) {
      paintJob(data.job);
      toast("Making the song");
    });
  }

  $("go").addEventListener("click", function () {
    var sheet = sheetValues();
    if (STATE.mode === "custom") {
      var style = $("style").value.trim(), lyrics = $("lyrics").value.trim();
      if (!style || !lyrics) { toast("A style and some lyrics are both needed", "bad"); return; }
      $("go").disabled = true;
      generate({ style: style, lyrics: lyrics, title: $("title").value.trim() || "song",
                 id: "song", cot: COT })
        .catch(function (error) { toast(error.message, "bad"); $("go").disabled = false; });
      return;
    }

    var idea = $("idea").value.trim();
    if (!idea) { toast("Describe the song in a line first", "bad"); return; }
    $("go").disabled = true;
    $("run").hidden = false;
    $("runTitle").textContent = "Writing the words";
    $("runSteps").innerHTML = '<li class="step" data-s="running"><span class="dot"></span>The writer is working</li>';
    $("runBar").style.width = "0%";
    $("cancel").hidden = true;
    runStarted = Date.now();
    stopClock();
    runTicker = setInterval(function () {
      $("runClock").textContent = clock((Date.now() - runStarted) / 1000);
    }, 1000);

    // One writer, one model: whatever the console has configured for llama.cpp.
    post("/api/muse", { idea: idea, backend: "llamacpp", structure: STRUCTURE, sheet: sheet })
      .then(function (brief) {
        return generate({ style: brief.style, lyrics: brief.lyrics,
                          title: brief.title || "song", id: "song",
                          cover: brief.cover || "", cot: COT });
      })
      .catch(function (error) {
        stopClock();
        $("run").hidden = true;
        $("go").disabled = false;
        toast(error.message, "bad");
      });
  });

  /* ------------------------------------------------- kept on this device */

  // Cache Storage and service workers need a secure origin, which a console on
  // plain HTTP over your own network is not. IndexedDB has no such rule, so a
  // saved song lives there as a blob and plays from memory rather than the wire.
  var DB_NAME = "yue2-songs", STORE = "audio";
  var dbPromise = null;

  function db() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise(function (resolve, reject) {
      if (!window.indexedDB) return reject(new Error("This browser keeps nothing offline"));
      var request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = function () {
        if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE);
      };
      request.onsuccess = function () { resolve(request.result); };
      request.onerror = function () { reject(request.error); };
    });
    return dbPromise;
  }

  function store(mode, work) {
    return db().then(function (handle) {
      return new Promise(function (resolve, reject) {
        var transaction = handle.transaction(STORE, mode);
        var request = work(transaction.objectStore(STORE));
        request.onsuccess = function () { resolve(request.result); };
        request.onerror = function () { reject(request.error); };
      });
    });
  }

  function savedBlob(name) { return store("readonly", function (s) { return s.get(name); }); }
  function savedNames() { return store("readonly", function (s) { return s.getAllKeys(); }); }
  function forget(name) { return store("readwrite", function (s) { return s.delete(name); }); }

  function keep(name, onProgress) {
    var url = "/api/library/" + encodeURIComponent(name) + "/audio";
    return fetch(url).then(function (response) {
      if (!response.ok) throw new Error("The console would not hand over the audio");
      var total = parseInt(response.headers.get("Content-Length") || "0", 10);
      if (!response.body || !window.ReadableStream) return response.blob();
      // Report progress where the browser allows it; a big song is a slow save.
      var reader = response.body.getReader(), chunks = [], done = 0;
      return (function pump() {
        return reader.read().then(function (piece) {
          if (piece.done) return new Blob(chunks, { type: "audio/flac" });
          chunks.push(piece.value);
          done += piece.value.length;
          if (onProgress && total) onProgress(done / total);
          return pump();
        });
      })();
    }).then(function (blob) {
      return store("readwrite", function (s) { return s.put(blob, name); }).then(function () { return blob; });
    });
  }

  function refreshSaved() {
    return savedNames().then(function (names) {
      STATE.saved = names || [];
      markSaved();
      return STATE.saved;
    }).catch(function () { STATE.saved = []; });
  }

  function isSaved(name) { return (STATE.saved || []).indexOf(name) >= 0; }

  function markSaved() {
    Array.prototype.forEach.call(document.querySelectorAll("[data-take]"), function (row) {
      row.dataset.saved = isSaved(row.dataset.take) ? "1" : "0";
    });
  }

  /* ------------------------------------------------------------- library */

  function refreshLibrary() {
    return api("/api/library").then(function (data) {
      STATE.takes = data.takes || [];
      var list = $("takes");
      $("takesEmpty").hidden = STATE.takes.length > 0;
      list.innerHTML = STATE.takes.map(function (take) {
        var art = take.has_cover
          ? '<img src="/api/library/' + encodeURIComponent(take.name) + '/cover" alt="" data-open="1" />'
          : '<div class="noart" data-open="1">♪</div>';
        return '<div class="take" data-take="' + escape(take.name) + '" data-saved="0">' + art +
          '<div class="who"><strong>' + escape(take.title) + "</strong>" +
          "<small>" + escape((take.style || "").split(",")[0]) + "</small></div>" +
          '<span class="len">' + clock(take.seconds) + "</span>" +
          '<span class="kept" aria-label="Saved on this device">●</span></div>';
      }).join("");
      markPlaying();
      markSaved();
    }).catch(function () {});
  }

  function markPlaying() {
    Array.prototype.forEach.call(document.querySelectorAll(".take"), function (row) {
      row.classList.toggle("is-playing", row.dataset.take === STATE.playing);
    });
  }

  $("takes").addEventListener("click", function (event) {
    var row = event.target.closest("[data-take]");
    if (!row) return;
    if (event.target.closest("[data-open]")) openSong(row.dataset.take);
    else play(row.dataset.take);
  });

  /* ---------------------------------------------------------- song view */

  function openSong(name) {
    var take = STATE.takes.filter(function (t) { return t.name === name; })[0];
    if (!take) return;
    STATE.viewing = name;

    if (take.has_cover) {
      $("songArt").src = "/api/library/" + encodeURIComponent(name) + "/cover";
      $("songArt").hidden = false;
      $("songNoArt").hidden = true;
    } else {
      $("songArt").hidden = true;
      $("songNoArt").hidden = false;
    }
    $("songTitle").textContent = take.title;
    var made = take.created ? new Date(take.created * 1000) : null;
    $("songMeta").textContent = [
      clock(take.seconds),
      take.seed !== undefined && take.seed !== null ? "seed " + take.seed : "",
      made ? made.toLocaleDateString() + " " + made.toLocaleTimeString().slice(0, 5) : "",
      take.engine === "gguf" ? "audio.cpp" : ""
    ].filter(Boolean).join("  ·  ");

    $("songStyle").textContent = take.style || "—";
    $("songLyrics").textContent = take.lyrics || "—";
    $("songCoverWrap").hidden = !take.cover_prompt;
    $("songCover").textContent = take.cover_prompt || "";
    buildKaraoke(take);
    loadTiming(name);

    paintPlayGlyph(STATE.playing === name && !audio.paused);
    paintSaveKey();
    savedBlob(name).then(function (blob) {
      if (!blob) return;
      $("songDownload").href = URL.createObjectURL(blob);
      $("songDownload").setAttribute("download", name + ".flac");
    }).catch(function () {});
    show("song");
    if (!drawing) draw();
  }

  /* ------------------------------------------------------------ karaoke */

  // Nothing in the pipeline records when a line is sung, so the timing here is
  // an estimate: each sung line is given a share of the song in proportion to
  // its length, with section tags taking none. It follows the song, it does not
  // know it. Real timing would need the audio aligned against the words.
  var lines = [], timed = null;

  function loadTiming(name) {
    // Real times if the take has been listened to; the even spacing otherwise.
    timed = null;
    return api("/api/library/" + encodeURIComponent(name) + "/timing").then(function (data) {
      if (!data || !data.lines || !data.lines.length) return;
      timed = {};
      data.lines.forEach(function (entry) { timed[entry.index] = entry; });
      applyTiming(data.duration);
      paintSyncKey(data);
    }).catch(function () { paintSyncKey(null); });
  }

  function applyTiming(duration) {
    if (!timed) return;
    lines.forEach(function (line, index) {
      var entry = timed[index];
      if (!entry || !duration) return;
      line.from = entry.start / duration;
      line.to = entry.end / duration;
    });
  }

  function paintSyncKey(data) {
    var key = $("songSync");
    if (data) {
      key.textContent = "Words timed to the song";
      key.dataset.done = "1";
      $("songSyncNote").textContent = data.lines_matched + " of " + data.lines_total +
        " lines were heard in the audio; the rest sit between them.";
    } else {
      key.textContent = "Time the words to the song";
      key.dataset.done = "0";
      $("songSyncNote").textContent = "Without this the words are spaced evenly, which drifts.";
    }
  }

  function buildKaraoke(take) {
    var raw = (take.lyrics || "").split("\n");
    var weights = [], total = 0;
    lines = raw.map(function (text) {
      var trimmed = text.trim();
      var tag = /^\[.*\]$/.test(trimmed);
      var weight = tag || !trimmed ? 0 : Math.max(8, trimmed.length);
      weights.push(weight);
      total += weight;
      return { text: trimmed, tag: tag, from: 0, to: 0 };
    });
    var running = 0;
    lines.forEach(function (line, index) {
      line.from = total ? running / total : 0;
      running += weights[index];
      line.to = total ? running / total : 0;
    });
    applyTiming(take.seconds);
    $("karaokeLines").innerHTML = lines.map(function (line, index) {
      if (!line.text) return "";
      return '<p class="kline' + (line.tag ? " is-tag" : "") + '" data-line="' + index + '">' +
        escape(line.text) + "</p>";
    }).join("");
    paintKaraoke(0);
  }

  function paintKaraoke(share) {
    if (!lines.length || $("karaoke").hidden) return;
    var current = -1;
    for (var i = 0; i < lines.length; i++) {
      if (!lines[i].tag && lines[i].text && share >= lines[i].from && share < lines[i].to) { current = i; break; }
    }
    var box = $("karaokeLines");
    Array.prototype.forEach.call(box.children, function (node) {
      var index = parseInt(node.dataset.line, 10);
      node.classList.toggle("is-now", index === current);
      node.classList.toggle("is-past", index < current);
    });
    var active = current >= 0 ? box.querySelector('[data-line="' + current + '"]') : null;
    if (active) {
      // Hold the sung line in the middle of the sleeve.
      box.style.transform = "translateY(" + (box.parentNode.clientHeight / 2 -
        active.offsetTop - active.offsetHeight / 2) + "px)";
    }
  }

  $("karaokeToggle").addEventListener("click", function () {
    var on = $("karaoke").hidden;
    $("karaoke").hidden = !on;
    this.setAttribute("aria-pressed", on ? "true" : "false");
    this.textContent = on ? "Cover" : "Words";
    if (on && audio.duration) paintKaraoke(audio.currentTime / audio.duration);
  });

  function paintSaveKey() {
    var name = STATE.viewing;
    if (!name) return;
    var kept = isSaved(name);
    $("songSave").textContent = kept ? "Remove from this device" : "Save on this device";
    $("songSaveNote").textContent = kept
      ? "Kept here, so it plays without the console and survives going out of range."
      : "Keeps the audio in this browser so it plays without the network.";
    $("songDownload").hidden = !kept;
  }

  $("songSync").addEventListener("click", function () {
    var name = STATE.viewing;
    if (!name) return;
    $("songSync").disabled = true;
    $("songSync").textContent = "Listening to the take…";
    api("/api/library/" + encodeURIComponent(name) + "/timing", { method: "POST" })
      .then(function () { toast("Listening to the take — this takes a moment"); })
      .catch(function (error) {
        $("songSync").disabled = false;
        paintSyncKey(null);
        toast(error.message, "bad");
      });
  });

  $("songBack").addEventListener("click", function () { show("listen"); });
  $("songPlay").addEventListener("click", function () {
    if (STATE.playing === STATE.viewing) {
      if (audio.paused) audio.play(); else audio.pause();
    } else {
      play(STATE.viewing);
    }
  });

  $("songSave").addEventListener("click", function () {
    var name = STATE.viewing;
    if (!name) return;
    if (isSaved(name)) {
      forget(name).then(refreshSaved).then(function () {
        paintSaveKey();
        toast("Removed from this device");
      }).catch(function (error) { toast(error.message, "bad"); });
      return;
    }
    $("songSave").disabled = true;
    $("songSave").textContent = "Saving…";
    keep(name, function (share) { $("songSave").textContent = "Saving… " + Math.round(share * 100) + "%"; })
      .then(function (blob) {
        var url = URL.createObjectURL(blob);
        $("songDownload").href = url;
        $("songDownload").setAttribute("download", name + ".flac");
        return refreshSaved();
      })
      .then(function () {
        $("songSave").disabled = false;
        paintSaveKey();
        toast("Saved on this device");
      })
      .catch(function (error) {
        $("songSave").disabled = false;
        paintSaveKey();
        toast(error.message, "bad");
      });
  });

  /* ------------------------------------------------------- visualisation */

  // A live meter off the audio element. If the browser will not give us an
  // analyser, the song still plays -- the canvas simply stays empty.
  var context = null, analyser = null, source = null, bins = null, drawing = false, tried = false;

  function listen() {
    if (tried) return;
    tried = true;
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    try {
      context = new Ctx();
      if (context.state === "suspended" && context.resume) context.resume();
      // Routing the element through Web Audio means a context that will not
      // run is a silent song. If it is not running by now, leave the audio
      // alone and do without the picture.
      if (context.state !== "running") { context = null; return; }
      source = context.createMediaElementSource(audio);
      analyser = context.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.78;
      source.connect(analyser);
      analyser.connect(context.destination);
      bins = new Uint8Array(analyser.frequencyBinCount);
    } catch (error) {
      analyser = null;
    }
  }

  function draw() {
    var canvas = $("viz");
    if (!analyser || !canvas || $("screen-song").className.indexOf("is-on") < 0) {
      drawing = false;
      return;
    }
    drawing = true;
    var width = canvas.clientWidth || 320;
    if (canvas.width !== width) canvas.width = width;
    var pen = canvas.getContext("2d");
    var height = canvas.height;
    analyser.getByteFrequencyData(bins);
    pen.clearRect(0, 0, width, height);
    var count = Math.min(bins.length, 40);
    var slot = width / count;
    for (var i = 0; i < count; i++) {
      var value = bins[i] / 255;
      var tall = Math.max(2, value * height);
      pen.fillStyle = "rgba(232, 163, 61, " + (0.35 + value * 0.65) + ")";
      pen.fillRect(i * slot + 1, height - tall, slot - 2, tall);
    }
    requestAnimationFrame(draw);
  }

  /* -------------------------------------------------------------- player */

  function play(name) {
    var take = STATE.takes.filter(function (t) { return t.name === name; })[0];
    if (!take) return;
    STATE.playing = name;
    $("now").hidden = false;
    $("nowTitle").textContent = take.title;
    $("nowStyle").textContent = (take.style || "").split(",").slice(0, 2).join(", ");
    if (take.has_cover) {
      $("nowArt").src = "/api/library/" + encodeURIComponent(name) + "/cover";
      $("nowArt").hidden = false;
      $("nowNoArt").hidden = true;
    } else {
      $("nowArt").hidden = true;
      $("nowNoArt").hidden = false;
    }
    // Playing is a gesture, which is when a browser will allow an analyser.
    listen();
    if (context && context.state === "suspended") context.resume();

    var network = "/api/library/" + encodeURIComponent(name) + "/audio";
    var start = function (src, kept) {
      if (STATE.source) URL.revokeObjectURL(STATE.source);
      STATE.source = kept ? src : null;
      audio.src = src;
      audio.play().catch(function () { toast("Tap play to start it"); });
      if (!drawing) draw();
    };
    savedBlob(name).then(function (blob) {
      start(blob ? URL.createObjectURL(blob) : network, !!blob);
    }).catch(function () { start(network, false); });
    markPlaying();
  }

  function paintPlayGlyph(playing) {
    $("playPause").dataset.playing = playing ? "1" : "0";
    $("songPlay").textContent = playing ? "Pause" : "Play";
  }

  $("playPause").addEventListener("click", function () {
    if (!STATE.playing) {
      if (STATE.takes.length) { play(STATE.takes[0].name); show("listen"); }
      else toast("Nothing to play yet", "bad");
      return;
    }
    if (audio.paused) audio.play(); else audio.pause();
  });

  audio.addEventListener("play", function () { paintPlayGlyph(true); if (!drawing) draw(); });
  audio.addEventListener("pause", function () { paintPlayGlyph(false); });
  function paintTime() {
    var share = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
    if (STATE.viewing === STATE.playing) paintKaraoke(share / 100);
    $("seekFill").style.width = share + "%";
    $("seekKnob").style.left = share + "%";
    $("nowTime").textContent = clock(audio.currentTime);
    $("nowLeft").textContent = audio.duration ? "-" + clock(audio.duration - audio.currentTime) : "";
  }
  audio.addEventListener("timeupdate", function () { if (!dragging) paintTime(); });
  audio.addEventListener("loadedmetadata", paintTime);
  audio.addEventListener("ended", function () {
    // Straight on to the next one down the list, like any music player.
    var index = STATE.takes.map(function (t) { return t.name; }).indexOf(STATE.playing);
    if (index >= 0 && index + 1 < STATE.takes.length) play(STATE.takes[index + 1].name);
  });

  // Dragging the line scrubs; the audio only moves when you let go, so a slow
  // seek does not stutter the sound on the way.
  var dragging = false, pending = 0;

  function seekFromEvent(event) {
    var box = $("seek").getBoundingClientRect();
    var point = event.touches ? event.touches[0].clientX : event.clientX;
    var share = Math.min(1, Math.max(0, (point - box.left) / box.width));
    pending = share * (audio.duration || 0);
    $("seekFill").style.width = share * 100 + "%";
    $("seekKnob").style.left = share * 100 + "%";
    $("nowTime").textContent = clock(pending);
    $("nowLeft").textContent = audio.duration ? "-" + clock(audio.duration - pending) : "";
  }

  function startDrag(event) {
    if (!audio.duration) return;
    dragging = true;
    seekFromEvent(event);
    event.preventDefault();
  }

  function moveDrag(event) { if (dragging) { seekFromEvent(event); event.preventDefault(); } }

  function endDrag() {
    if (!dragging) return;
    dragging = false;
    audio.currentTime = pending;
  }

  $("seek").addEventListener("mousedown", startDrag);
  $("seek").addEventListener("touchstart", startDrag, { passive: false });
  document.addEventListener("mousemove", moveDrag);
  document.addEventListener("touchmove", moveDrag, { passive: false });
  document.addEventListener("mouseup", endDrag);
  document.addEventListener("touchend", endDrag);

  /* ---------------------------------------------------------------- live */

  function connect() {
    var source = new EventSource("/api/events");
    source.addEventListener("job", function (event) {
      var job = JSON.parse(event.data);
      if (!STATE.job || STATE.job.id === job.id) paintJob(job);
      if (job.state === "done") {
        stopClock();
        $("run").hidden = true;
        $("go").disabled = false;
        toast("Song finished: " + job.title, "good");
        refreshLibrary().then(function () { if (job.take) play(job.take); });
        show("listen");
      }
    });
    source.addEventListener("library", function () { refreshLibrary(); });
    source.addEventListener("timing", function (event) {
      var data = JSON.parse(event.data);
      if (data.error) { toast(data.error, "bad"); }
      if (data.busy) { $("songSync").textContent = data.busy + "…"; return; }
      $("songSync").disabled = false;
      if (data.done && !data.error && data.take === STATE.viewing) {
        loadTiming(data.take).then(function () { toast("The words now follow the song"); });
      } else if (!data.busy) {
        paintSyncKey(null);
      }
    });
    source.addEventListener("engine", function (event) {
      var engine = JSON.parse(event.data);
      $("engineNote").textContent = engine.status === "error"
        ? engine.detail : (engine.status === "loading" ? "Loading the model…" : "");
    });
    source.onerror = function () { /* EventSource retries on its own */ };
  }

  /* ---------------------------------------------------------------- boot */

  try { setMode(localStorage.getItem("yue2m.mode") === "custom" ? "custom" : "simple"); }
  catch (error) { setMode("simple"); }

  api("/api/vocabulary").then(function (data) {
    if (data && data.order && data.order.length) buildSheet(data);
  }).catch(function () {});

  api("/api/state").then(function (data) {
    var live = (data.jobs || []).filter(function (j) {
      return j.state === "running" || j.state === "queued";
    })[0];
    if (live) paintJob(live);
    connect();
  }).catch(function (error) { toast("Cannot reach the console: " + error.message, "bad"); });

  refreshLibrary().then(refreshSaved);
})();

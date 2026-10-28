// Standalone editor; all serialized output changes pass through onSelection.
export function validRange(start, end, duration) {
    return Number.isFinite(start) && Number.isFinite(end) && start >= 0 && start < end && end <= duration;
}

export function createCutAudioEditor({ request, apiUrl = (s) => s, onFile, onSelection }) {
    const panel = document.createElement("div");
    panel.className = "jr-cut-audio";
    if (!document.getElementById("jr-cut-audio-style")) {
        const style = document.createElement("style"); style.id = "jr-cut-audio-style";
        style.textContent = `
        .jr-cut-audio{box-sizing:border-box;background:#171d26;color:#e3edf5;padding:14px;font:13px system-ui;width:100%;height:100%;overflow:auto;border-radius:10px}
        .jr-cut-audio *{box-sizing:border-box}.jr-cut-audio .row{display:flex;align-items:center;gap:8px;margin:10px 0;flex-wrap:wrap}
        .jr-cut-audio button,.jr-cut-audio select,.jr-cut-audio input[type=number]{background:#273443;color:#e3edf5;border:1px solid #48596e;border-radius:5px;padding:7px}
        .jr-cut-audio button{cursor:pointer}.jr-cut-audio button:disabled{opacity:.45;cursor:default}.jr-cut-audio select{width:100%}
        .jr-cut-audio a{color:#78d6df}.jr-cut-audio a:not([href]){opacity:.4;pointer-events:none}.jr-cut-audio canvas{width:100%;height:170px;touch-action:none;background:#101722;border-radius:6px;cursor:crosshair}
        .jr-cut-audio input[type=range]{accent-color:#62d5cf;min-width:60px}.jr-cut-audio .seek{flex:1}.jr-cut-audio .volume{width:100px}
        .jr-cut-audio .times label{flex:1}.jr-cut-audio input[type=number]{display:block;width:100%;margin-top:5px}.jr-cut-audio .primary{background:#196d68;border-color:#46bbb0;flex:1}
        .jr-cut-audio .status{white-space:pre-wrap;overflow-wrap:anywhere;color:#9fb4c6;min-height:38px}.jr-cut-audio .error{color:#ffaba4}.jr-cut-audio .hint{color:#8da6bb;font-size:11px}.jr-cut-audio .title{font-weight:650;letter-spacing:.8px}.jr-cut-audio [hidden]{display:none!important}
        `; document.head.append(style);
    }
    // Static markup only; filenames and server messages always use textContent.
    panel.innerHTML = `<div class="title">JR / CUT AUDIO</div>
      <div class="row"><button class="upload">Choose file to upload</button><button class="refresh">Refresh files</button><input class="file" type="file" accept=".mp3,.wav,.flac,.ogg,.opus,.m4a,.aac,.aif,.aiff,.wma" hidden></div>
      <select class="sources" aria-label="Source audio"><option value="">Choose existing audio…</option></select>
      <div class="row"><a class="original" download>Download original</a><span class="details hint"></span></div>
      <canvas tabindex="0" aria-label="Waveform: drag playhead or selection edges; arrow keys seek"></canvas>
      <div class="hint">Drag the white playhead or cyan selection edges · Arrow keys seek</div>
      <div class="row"><button class="play" disabled>▶ Play</button><input class="seek" type="range" min="0" max="1" step="0.001" value="0" aria-label="Playback position"><span class="clock">0.000 s</span></div>
      <div class="row"><label>Volume <input class="volume" type="range" min="0" max="1" step="0.01" value="0.8" aria-label="Playback volume"></label><button class="mode" disabled>Preview: source</button></div>
      <div class="row times"><label>Start (seconds)<input class="start" type="number" min="0" step="0.001" value="0"></label><label>End (seconds)<input class="end" type="number" min="0" step="0.001" value="0"></label></div>
      <div class="row"><button class="at-start" disabled>Cursor → start</button><button class="at-end" disabled>Cursor → end</button></div>
      <div class="row"><button class="cut primary" disabled>Cut & lock output</button><button class="unlock" disabled>Unlock / reselect</button><a class="download" download>Download cut WAV</a></div>
      <div class="status" role="status">Upload audio to begin. Cut locks the AUDIO sent downstream.</div>`;
    const q = (s) => panel.querySelector(s);
    const player = document.createElement("audio"); player.preload = "metadata"; player.volume = .8; player.hidden=true; panel.append(player);
    const canvas = q("canvas"), start = q(".start"), end = q(".end");
    let info = null, filename = "", locked = false, cutToken = "", clip = false;
    let busy = false, disposed = false, generation = 0, controller = new AbortController(), animation = 0;
    const status = (text, error = false) => { q(".status").textContent = text; q(".status").classList.toggle("error", error); };
    const url = (token, download = false) => apiUrl(`/jr-cut-audio/media/${token}.wav${download ? "?download=1" : ""}`);
    const absoluteTime = () => (clip ? Number(start.value) : 0) + (player.currentTime || 0);
    const refreshControls = () => {
        for (const s of [".upload", ".refresh", ".sources"]) q(s).disabled = busy;
        for (const s of [".cut", ".at-start", ".at-end"]) q(s).disabled = busy || !info || locked;
        start.disabled = end.disabled = busy || !info || locked;
        q(".unlock").disabled = busy || !locked; q(".mode").disabled = busy || !cutToken;
        q(".play").disabled = busy || !info; q(".seek").disabled = busy || !info;
        draw();
    };
    function draw() {
        if (disposed) return;
        const width = canvas.clientWidth || 480, height = 170, dpr = window.devicePixelRatio || 1;
        if (canvas.width !== Math.round(width*dpr) || canvas.height !== height*dpr) { canvas.width = Math.round(width*dpr); canvas.height = height*dpr; }
        const c = canvas.getContext("2d"); c.setTransform(dpr,0,0,dpr,0,0); c.clearRect(0,0,width,height);
        c.strokeStyle = "#263749"; c.lineWidth = 1;
        for (const y of [35,85,135]) { c.beginPath(); c.moveTo(0,y); c.lineTo(width,y); c.stroke(); }
        if (!info) { c.fillStyle="#8da6bb"; c.fillText("Upload a file to display its waveform",15,85); return; }
        const x1 = Number(start.value)/info.duration*width, x2 = Number(end.value)/info.duration*width;
        c.fillStyle = locked ? "#245d4744" : "#287b9744"; c.fillRect(x1,0,x2-x1,150);
        c.strokeStyle = "#66c7ce"; c.beginPath();
        info.peaks.forEach(([lo,hi],i) => { const x=i/info.peaks.length*width; c.moveTo(x,85-Math.max(-1,Math.min(1,hi))*58); c.lineTo(x,85-Math.max(-1,Math.min(1,lo))*58); }); c.stroke();
        c.strokeStyle = locked ? "#67e4a3" : "#6de4f1"; c.lineWidth = 2;
        for (const x of [x1,x2]) { c.beginPath(); c.moveTo(x,0); c.lineTo(x,150); c.stroke(); c.fillStyle=c.strokeStyle; c.fillRect(Math.max(0,Math.min(width-8,x-4)),0,8,16); }
        const cursor = absoluteTime()/info.duration*width;
        c.strokeStyle = "#ffffff"; c.beginPath(); c.moveTo(cursor,0); c.lineTo(cursor,150); c.stroke();
        c.font="10px system-ui"; c.fillStyle="#94aec3";
        for(let i=0;i<=4;i++){const label=(info.duration*i/4).toFixed(2)+"s"; c.fillText(label,Math.min(width-c.measureText(label).width,width*i/4),165);}
        q(".clock").textContent = `${absoluteTime().toFixed(3)} s`;
        q(".seek").value = String(player.currentTime || 0);
    }
    function media(token, isClip) {
        player.pause(); clip=isClip; player.src=url(token); player.load();
        q(".play").textContent="▶ Play";
        q(".mode").textContent = isClip ? "Preview: cut" : "Preview: source";
        q(".seek").max = String(isClip ? Number(end.value)-Number(start.value) : info.duration); draw();
    }
    async function json(path, options = {}) {
        const response = await request(path, {...options, signal:controller.signal});
        if (!response.ok) throw new Error((await response.text()).slice(0,300) || `HTTP ${response.status}`);
        return response.json();
    }
    const post = (path, data) => json(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    function addOption(name) {
        if (![...q(".sources").options].some(o=>o.value===name)) {const option=document.createElement("option"); option.value=name; option.textContent=name; q(".sources").append(option);}
        q(".sources").value=name;
    }
    async function files() {
        try { const data=await json("/jr-cut-audio/files"); if(disposed)return;
            q(".sources").replaceChildren(new Option("Choose existing audio…",""));
            for(const name of data.files) { const option=new Option(name,name); q(".sources").append(option); }
            if(filename)addOption(filename);
        } catch(e) {if(e.name!=="AbortError"&&!disposed)status(e.message,true);}
    }
    async function cut(saved = null) {
        if(!info)return;
        const from=saved ? saved.start_sample/saved.sample_rate : Number(start.value);
        const to=saved ? saved.end_sample/saved.sample_rate : Number(end.value);
        if(!validRange(from,to,info.duration))throw new Error("Use 0 ≤ start < end ≤ audio duration.");
        if(saved && (saved.audio!==filename || saved.source_id!==info.source_id || saved.sample_rate!==info.sample_rate || saved.source_samples!==info.samples)) throw new Error("Saved selection no longer matches the source. Unlock and Cut again.");
        const ticket=generation;
        const result=await post("/jr-cut-audio/cut",{audio:filename,start:from,end:to,source_id:info.source_id});
        if(disposed||ticket!==generation)return;
        start.value=String(result.start_seconds); end.value=String(result.end_seconds);
        cutToken=result.media_token; locked=true; onSelection(result.selection);
        q(".download").href=url(cutToken,true); media(cutToken,true);
        status(`LOCKED · ${result.duration.toFixed(6)} s · ${info.sample_rate} Hz · ${info.channels} channels\nAUDIO output and downloaded WAV contain the same samples.`);
    }
    async function restore(name, selection = "") {
        const ticket=++generation; controller.abort(); controller=new AbortController();
        player.pause(); player.removeAttribute("src"); player.load();
        filename=name || ""; info=null; cutToken=""; clip=false; locked=false; start.value=end.value="0";
        q(".details").textContent="";q(".clock").textContent="0.000 s";q(".play").textContent="▶ Play";
        q(".download").removeAttribute("href"); q(".original").removeAttribute("href");
        if(!filename){q(".sources").value="";busy=false;status("Upload audio to begin. Cut locks the AUDIO sent downstream.");refreshControls();return;}
        addOption(filename); busy=true; refreshControls(); status("Decoding audio and building waveform…");
        try {
            const data=await post("/jr-cut-audio/info",{audio:filename}); if(disposed||ticket!==generation)return;
            info=data; start.value="0"; end.value=String(info.duration);
            q(".details").textContent=`${info.duration.toFixed(3)} s · ${info.sample_rate} Hz · ${info.channels} ch`;
            q(".original").href=apiUrl(`/jr-cut-audio/original?audio=${encodeURIComponent(filename)}`);
            media(info.media_token,false);
            if(selection) { locked=true; await cut(JSON.parse(selection)); }
            else status("Set start/end seconds or drag the edges, then Cut & lock output.");
        } catch(e) { if(e.name!=="AbortError"&&!disposed&&ticket===generation)status(e.message,true); }
        finally { if(!disposed&&ticket===generation){busy=false;refreshControls();} }
    }
    async function action(fn) {
        const ticket=generation; busy=true;refreshControls();
        try { await fn(); } catch(e) {if(e.name!=="AbortError"&&!disposed)status(e.message,true);}
        finally { if(!disposed&&ticket===generation){busy=false;refreshControls();} }
    }
    q(".upload").onclick=()=>q(".file").click();
    q(".file").onchange=()=>action(async()=>{
        const file=q(".file").files[0]; q(".file").value=""; if(!file)return;
        if(file.size>512*1024*1024)throw new Error("Maximum source file size is 512 MiB.");
        status("Uploading to your ComfyUI input directory…");
        const form=new FormData();form.append("image",file);form.append("type","input");form.append("subfolder","jr_cut_audio");
        const data=await json("/upload/image",{method:"POST",body:form});if(disposed)return;
        const name=(data.subfolder ? data.subfolder+"/" : "")+data.name;
        onFile(name);onSelection("");await restore(name);
    });
    q(".refresh").onclick=()=>action(files);
    q(".sources").onchange=()=>{onFile(q(".sources").value);onSelection("");restore(q(".sources").value);};
    q(".cut").onclick=()=>action(()=>cut());
    q(".unlock").onclick=()=>{locked=false;cutToken="";onSelection("");q(".download").removeAttribute("href");if(info)media(info.media_token,false);status("Unlocked. Press Cut to commit a new AUDIO output.");refreshControls();};
    q(".play").onclick=async()=>{try {if(player.paused)await player.play();else player.pause();}catch(e){status(`Playback unavailable: ${e.message}`,true);}};
    q(".volume").oninput=()=>{player.volume=Number(q(".volume").value);};
    q(".seek").oninput=()=>{player.currentTime=Number(q(".seek").value);draw();};
    q(".mode").onclick=()=>media(clip?info.media_token:cutToken,!clip);
    const seek=(seconds)=>{const offset=clip?Number(start.value):0;player.currentTime=Math.max(0,Math.min(clip?Number(end.value)-offset:info.duration,seconds-offset));draw();};
    q(".at-start").onclick=()=>{start.value=String(absoluteTime());draw();};
    q(".at-end").onclick=()=>{end.value=String(absoluteTime());draw();};
    start.oninput=end.oninput=()=>draw();
    let drag=null;
    const pointerTime=(e)=>{const r=canvas.getBoundingClientRect();return Math.max(0,Math.min(info.duration,(e.clientX-r.left)/r.width*info.duration));};
    canvas.onpointerdown=(e)=>{
        if(!info||busy)return;e.preventDefault();canvas.focus();canvas.setPointerCapture(e.pointerId);
        const t=pointerTime(e), threshold=12/canvas.getBoundingClientRect().width*info.duration;
        drag=!locked&&Math.abs(t-Number(start.value))<threshold?"start":!locked&&Math.abs(t-Number(end.value))<threshold?"end":"seek";
        canvas.onpointermove(e);
    };
    canvas.onpointermove=(e)=>{if(!drag||!info)return;const t=pointerTime(e);if(drag==="seek")seek(t);else{
        const bounded=drag==="start"?Math.max(0,Math.min(t,Number(end.value)-1/info.sample_rate)):Math.min(info.duration,Math.max(t,Number(start.value)+1/info.sample_rate));
        (drag==="start"?start:end).value=String(bounded);draw();}};
    canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
    canvas.onkeydown=(e)=>{if(info&&["ArrowLeft","ArrowRight"].includes(e.key)){e.preventDefault();seek(absoluteTime()+(e.key==="ArrowRight"?1:-1)*(e.shiftKey?1:.1));}};
    const tick=()=>{draw();if(!player.paused&&!disposed)animation=requestAnimationFrame(tick);};
    player.onplay=()=>{q(".play").textContent="❚❚ Pause";cancelAnimationFrame(animation);tick();};
    player.onpause=player.onended=()=>{q(".play").textContent="▶ Play";cancelAnimationFrame(animation);draw();};
    player.ontimeupdate=()=>draw();
    for(const event of ["pointerdown","wheel","keydown"])panel.addEventListener(event,e=>e.stopPropagation());
    const observer=new ResizeObserver(draw);observer.observe(canvas);refreshControls();
    // File listing on explicit refresh avoids racing restoration of several nodes.
    return {panel,restore,destroy(){disposed=true;generation++;controller.abort();cancelAnimationFrame(animation);observer.disconnect();player.pause();player.removeAttribute("src");player.load();}};
}

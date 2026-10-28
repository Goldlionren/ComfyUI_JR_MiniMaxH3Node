import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createCutAudioEditor } from "./cut_audio_editor.mjs";

const instances = new WeakMap();
app.registerExtension({
    name: "JR.CutAudio",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "JR_CutAudio") return;
        const created=nodeType.prototype.onNodeCreated, configured=nodeType.prototype.onConfigure, removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated=function() {
            const result=created?.apply(this,arguments);
            const audio=this.widgets.find(w=>w.name==="audio"), selection=this.widgets.find(w=>w.name==="selection");
            for(const widget of [audio,selection]) {
                widget.type="jr-hidden-state";widget.computeSize=()=>[0,-4];widget.serializeValue=()=>widget.value;
                if(widget.inputEl)widget.inputEl.style.display="none";
            }
            const set=(widget,value)=>{if(widget.value!==value){widget.value=value;this.graph?.change();this.setDirtyCanvas(true,true);}};
            const editor=createCutAudioEditor({request:(...args)=>api.fetchApi(...args),apiUrl:path=>api.apiURL(path),onFile:v=>set(audio,v),onSelection:v=>set(selection,v)});
            let signature=null;
            const sync=()=>{const next=JSON.stringify([audio.value,selection.value]);if(next!==signature){signature=next;editor.restore(audio.value,selection.value);}};
            instances.set(this,{editor,sync});
            this.addDOMWidget("jr_cut_audio_editor","jr-cut-audio",editor.panel,{serialize:false,hideOnZoom:false,getMinHeight:()=>630});
            this.setSize([Math.max(600,this.size[0]),Math.max(700,this.size[1])]);
            return result;
        };
        nodeType.prototype.onConfigure=function(){const result=configured?.apply(this,arguments);instances.get(this)?.sync();return result;};
        nodeType.prototype.onRemoved=function(){instances.get(this)?.editor.destroy();instances.delete(this);return removed?.apply(this,arguments);};
    },
    loadedGraphNode(node){instances.get(node)?.sync();},
});

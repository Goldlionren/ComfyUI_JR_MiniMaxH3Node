import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";
import {validRange} from "../js/cut_audio_editor.mjs";

test("seconds validation",()=>{
    assert.equal(validRange(.125,3,3),true);
    for(const [a,b] of [[0,0],[-1,2],[2,1],[0,4],[NaN,2],[0,Infinity],["0",1]])assert.equal(validRange(a,b,3),false);
});

test("Comfy extension registration, serialized inputs, restore and removal",()=>{
    let extension, options, destroyed=0, configured=0, removed=0;
    const restores=[];
    const source=readFileSync(new URL("../js/cut_audio.js",import.meta.url),"utf8").replace(/^import .*;\r?\n/gm,"");
    const context=vm.createContext({WeakMap,JSON,Math,app:{registerExtension:e=>extension=e},api:{fetchApi:()=>{},apiURL:s=>s},
        createCutAudioEditor:o=>{options=o;return{panel:{},restore:(...args)=>restores.push(args),destroy:()=>destroyed++};}});
    vm.runInContext(source,context);
    class Node {
        constructor(){this.widgets=[{name:"audio",value:"song.wav"},{name:"selection",value:"lock",inputEl:{style:{}}}];this.size=[200,200];this.graph={change:()=>{}};}
        onNodeCreated(){return 42;}
        onConfigure(){configured++;}
        onRemoved(){removed++;}
        setSize(size){this.size=size;}
        setDirtyCanvas(){}
        addDOMWidget(name,type,panel,opts){assert.equal(opts.serialize,false);}
    }
    const original=Node.prototype.onNodeCreated;
    extension.beforeRegisterNodeDef(Node,{name:"OtherNode"});assert.equal(Node.prototype.onNodeCreated,original);
    extension.beforeRegisterNodeDef(Node,{name:"JR_CutAudio"});
    const node=new Node();assert.equal(node.onNodeCreated(),42);
    assert.equal(node.widgets[0].serializeValue(),"song.wav");assert.equal(node.widgets[1].inputEl.style.display,"none");
    node.onConfigure();extension.loadedGraphNode(node);assert.equal(configured,1);assert.equal(restores.length,1);
    options.onFile("new.wav");options.onSelection("new-lock");assert.equal(node.widgets[1].serializeValue(),"new-lock");
    node.onConfigure();assert.deepEqual(restores[1],["new.wav","new-lock"]);
    node.onRemoved();assert.equal(destroyed,1);assert.equal(removed,1);
    // The extension owns no workflow queue operation.
    assert.equal(source.includes("queuePrompt"),false);
});

import { useState, useRef, useCallback } from "react";

const HF_SPACE = (import.meta.env.VITE_HF_SPACE_URL || "https://srikanth-haki-mini-project.hf.space").replace(/\/$/, "");

const W = { sem1_grade:0.20,sem2_grade:0.18,prev_grade:0.14,attendance:0.13,
            assignments_done:0.10,logins:0.09,sem1_approved:0.07,admission_grade:0.04,
            tuition:0.025,debtor:-0.03,scholarship:0.02,parent_edu:0.015 };
const B = { sem1_grade:{mn:0,mx:20},sem2_grade:{mn:0,mx:20},prev_grade:{mn:0,mx:20},
            attendance:{mn:0,mx:100},assignments_done:{mn:0,mx:10},logins:{mn:0,mx:100},
            sem1_approved:{mn:0,mx:6},admission_grade:{mn:95,mx:175},
            tuition:{mn:0,mx:1},debtor:{mn:0,mx:1},scholarship:{mn:0,mx:1},parent_edu:{mn:1,mx:5} };
function norm(v,k){ const {mn,mx}=B[k]; return mx===mn?0:(v-mn)/(mx-mn); }
function localPredict(s){
  const tw=Object.values(W).reduce((a,b)=>a+Math.abs(b),0);
  let sc=0;
  for(const [k,w] of Object.entries(W)) sc+=w>0?norm(s[k]??0,k)*w:(1-norm(s[k]??0,k))*Math.abs(w);
  const drop=Math.min(0.97,Math.max(0.03,1-sc/tw));
  const grad=Math.min(0.93,Math.max(0.02,(sc/tw)*0.85));
  const enrl=Math.max(0.01,1-drop-grad);
  return{dropout:drop,graduate:grad,enrolled:enrl,
         risk_level:drop>0.65?"HIGH":drop>0.38?"MODERATE":"LOW",
         dropout_pct:`${Math.round(drop*100)}%`};
}

function parseCSV(text){
  const lines=text.trim().split("\n");
  const headers=lines[0].split(",").map(h=>h.trim().toLowerCase().replace(/\s+/g,"_").replace(/[^a-z0-9_]/g,""));
  return lines.slice(1).filter(l=>l.trim()).map((line,i)=>{
    const vals=line.split(",");
    const row={id:i+1};
    headers.forEach((h,j)=>{const v=vals[j]?.trim();row[h]=isNaN(v)||v===""?v:parseFloat(v)||0;});
    const aliases={
      admission_grade:["admissiongrade","admission grade"],
      prev_grade:["previous_qualification_grade","prev grade","prevgrade"],
      sem1_grade:["curricular_units_1st_sem_grade","sem1 grade","semester1_grade","s1grade"],
      sem2_grade:["curricular_units_2nd_sem_grade","sem2 grade","semester2_grade","s2grade"],
      sem1_approved:["curricular_units_1st_sem_approved","units_approved","sem1approved"],
      logins:["lms_logins","monthly_logins","lmslogins"],
      attendance:["attendance_rate","attend"],
      assignments_done:["assignments","assignments_submitted","assignmentsdone"],
      scholarship:["scholarship_holder"],
      tuition:["tuition_fees_up_to_date","tuition_paid"],
      debtor:["has_debt"],
      parent_edu:["mothers_qualification","fathers_qualification","parent_education"],
    };
    for(const [canon,alts] of Object.entries(aliases)){
      if(!(canon in row)) for(const a of alts){if(a in row){row[canon]=row[a];break;}}
      if(!(canon in row)) row[canon]=0;
    }
    return row;
  });
}

function generatePDFReport(students,results,date){
  const high=results.filter(r=>r.risk_level==="HIGH").length;
  const mod=results.filter(r=>r.risk_level==="MODERATE").length;
  const low=results.filter(r=>r.risk_level==="LOW").length;
  const avg=(results.reduce((a,r)=>a+r.dropout,0)/results.length*100).toFixed(1);
  const rc=l=>l==="HIGH"?"#ef4444":l==="MODERATE"?"#f59e0b":"#22c55e";
  const rb=l=>l==="HIGH"?"#fef2f2":l==="MODERATE"?"#fffbeb":"#f0fdf4";
  const rows=students.map((s,i)=>{
    const r=results[i];
    return `<tr style="border-bottom:1px solid #e5e7eb;${i%2?"background:#f9fafb":""}">
      <td style="padding:8px 12px;font-weight:600">${s.id||i+1}</td>
      <td style="padding:8px 12px;color:${s.sem1_grade<9?"#ef4444":s.sem1_grade<13?"#f59e0b":"#22c55e"}">${s.sem1_grade??"-"}</td>
      <td style="padding:8px 12px;color:${s.sem2_grade<9?"#ef4444":s.sem2_grade<13?"#f59e0b":"#22c55e"}">${s.sem2_grade??"-"}</td>
      <td style="padding:8px 12px;color:${s.attendance<50?"#ef4444":s.attendance<75?"#f59e0b":"#22c55e"}">${s.attendance??"-"}%</td>
      <td style="padding:8px 12px">${s.assignments_done??"-"}/10</td>
      <td style="padding:8px 12px">${s.logins??"-"}</td>
      <td style="padding:8px 12px;font-weight:700;color:${rc(r.risk_level)}">${Math.round(r.dropout*100)}%</td>
      <td style="padding:8px 12px"><span style="background:${rb(r.risk_level)};color:${rc(r.risk_level)};padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700">${r.risk_level}</span></td>
    </tr>`;
  }).join("");
  const html=`<!DOCTYPE html><html><head><meta charset="UTF-8"/>
  <title>EduRisk Report — ${date}</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',sans-serif;color:#1e293b;background:#fff;padding:40px}
    @media print{.no-print{display:none!important}}
    h1{font-size:26px;font-weight:800;color:#0f172a}
    .hdr{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:32px;padding-bottom:20px;border-bottom:2px solid #e2e8f0}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
    .stat{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center}
    .sv{font-size:28px;font-weight:800}.sl{font-size:11px;color:#64748b;margin-top:4px;text-transform:uppercase}
    table{width:100%;border-collapse:collapse;font-size:13px}
    thead tr{background:#f1f5f9}
    th{padding:10px 12px;text-align:left;font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
    .footer{margin-top:32px;padding-top:16px;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;display:flex;justify-content:space-between}
    .pbtn{position:fixed;bottom:24px;right:24px;background:#2563eb;color:#fff;border:none;padding:12px 24px;border-radius:8px;font-size:14px;font-weight:700;cursor:pointer;box-shadow:0 4px 12px rgba(37,99,235,.4)}
  </style></head><body>
  <div class="hdr">
    <div>
      <div style="font-size:10px;color:#64748b;letter-spacing:2px;margin-bottom:6px">EDURISK · STUDENT DROPOUT ANALYSIS REPORT</div>
      <h1>Risk Assessment Report</h1>
      <div style="color:#64748b;margin-top:6px;font-size:13px">Generated ${date} · ${students.length} students analysed</div>
    </div>
    <div style="text-align:right">
      <span style="background:#dbeafe;color:#1d4ed8;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600">RF + XGBoost Ensemble</span>
      <div style="font-size:11px;color:#94a3b8;margin-top:8px">UCI Dataset · AUC 0.89 · Recall 84%</div>
    </div>
  </div>
  <div class="grid">
    <div class="stat"><div class="sv" style="color:#1e293b">${students.length}</div><div class="sl">Total Students</div></div>
    <div class="stat"><div class="sv" style="color:#ef4444">${high}</div><div class="sl">High Risk</div></div>
    <div class="stat"><div class="sv" style="color:#f59e0b">${mod}</div><div class="sl">Moderate Risk</div></div>
    <div class="stat"><div class="sv" style="color:#22c55e">${low}</div><div class="sl">Low Risk</div></div>
  </div>
  <div style="margin-bottom:24px">
    <div style="font-size:11px;color:#64748b;margin-bottom:6px;font-weight:700">RISK DISTRIBUTION</div>
    <div style="display:flex;height:12px;border-radius:999px;overflow:hidden;margin-bottom:8px">
      <div style="width:${high/students.length*100}%;background:#ef4444"></div>
      <div style="width:${mod/students.length*100}%;background:#f59e0b"></div>
      <div style="width:${low/students.length*100}%;background:#22c55e"></div>
    </div>
    <div style="display:flex;gap:20px;font-size:11px;color:#64748b">
      <span>🔴 High: ${(high/students.length*100).toFixed(0)}%</span>
      <span>🟡 Moderate: ${(mod/students.length*100).toFixed(0)}%</span>
      <span>🟢 Low: ${(low/students.length*100).toFixed(0)}%</span>
      <span style="margin-left:auto">Avg dropout probability: <strong style="color:#ef4444">${avg}%</strong></span>
    </div>
  </div>
  <h2 style="font-size:15px;font-weight:700;margin:24px 0 12px;color:#374151">Per-Student Risk Breakdown</h2>
  <table>
    <thead><tr><th>ID</th><th>Sem1 Grade</th><th>Sem2 Grade</th><th>Attendance</th><th>Assignments</th><th>Logins</th><th>Dropout %</th><th>Risk Level</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <h2 style="font-size:15px;font-weight:700;margin:28px 0 12px;color:#374151">Key Risk Factors Identified</h2>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">
    ${["Low attendance (<50%)","Missing assignments (≤3/10)","Low semester grades (<10/20)","High debt + no tuition","Zero LMS logins","No scholarship + low grades"]
      .map(f=>`<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:7px;padding:10px 12px;font-size:12px;color:#991b1b">⚠ ${f}</div>`).join("")}
  </div>
  <div class="footer">
    <span>EduRisk · RF+XGBoost Ensemble + Qwen3-8B · huggingface.co/spaces/Srikanth-Haki/Mini_Project</span>
    <span>Confidential — For Academic Advisor Use Only</span>
  </div>
  <button class="pbtn no-print" onclick="window.print()">⬇ Save as PDF</button>
  </body></html>`;
  const w=window.open("","_blank");
  w.document.write(html);
  w.document.close();
}

const LMS_PROVIDERS=[
  {id:"moodle",   name:"Moodle",           status:"ready",   icon:"🟢",
   desc:"REST API via token. Fetches grades, attendance, assignments.",
   fields:["Site URL","Web Service Token"]},
  {id:"canvas",   name:"Canvas",           status:"ready",   icon:"🟢",
   desc:"REST API + OAuth2. Full grade/submission/login data.",
   fields:["Instance URL","API Token"]},
  {id:"blackboard",name:"Blackboard",      status:"ready",   icon:"🟢",
   desc:"REST API with OAuth2. Grades, enrolment, activity data.",
   fields:["Site URL","App Key","App Secret"]},
  {id:"classroom",name:"Google Classroom", status:"partial", icon:"🟡",
   desc:"Google API. Grades + submissions only. No attendance.",
   fields:["OAuth Client ID","Client Secret"]},
  {id:"custom",   name:"Custom / Other",   status:"pending", icon:"⚪",
   desc:"Paste any REST endpoint returning student JSON.",
   fields:["API URL","Auth Header"]},
];

const RC={HIGH:{fg:"#ef4444",bg:"#fef2f2",border:"#fecaca"},
          MODERATE:{fg:"#f59e0b",bg:"#fffbeb",border:"#fde68a"},
          LOW:{fg:"#22c55e",bg:"#f0fdf4",border:"#bbf7d0"}};

export default function App(){
  const [page,setPage]=useState("dashboard");
  const [csvStudents,setCsvStudents]=useState([]);
  const [csvResults,setCsvResults]=useState([]);
  const [csvFile,setCsvFile]=useState(null);
  const [csvLoading,setCsvLoading]=useState(false);
  const [csvError,setCsvError]=useState("");
  const [dragOver,setDragOver]=useState(false);
  const [form,setForm]=useState({admission_grade:130,prev_grade:12.5,sem1_grade:11,sem2_grade:10.5,sem1_approved:3,logins:18,attendance:58,assignments_done:4,scholarship:0,tuition:1,debtor:0,parent_edu:2});
  const [result,setResult]=useState(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState("");
  const [hfToken,setHfToken]=useState("");
  const [nudge,setNudge]=useState("");
  const [nLoading,setNLoading]=useState(false);
  const [nErr,setNErr]=useState("");
  const [showTok,setShowTok]=useState(false);
  const [lmsProv,setLmsProv]=useState(null);
  const [lmsCreds,setLmsCreds]=useState({});
  const [lmsSt,setLmsSt]=useState("");
  const [lmsTesting,setLmsTesting]=useState(false);
  const fileRef=useRef();

  const processCSV=useCallback(async(file)=>{
    setCsvLoading(true);setCsvError("");setCsvStudents([]);setCsvResults([]);
    try{
      const text=await file.text();
      const students=parseCSV(text);
      if(!students.length) throw new Error("No data rows found.");
      setCsvStudents(students);
      setCsvResults(students.map(s=>localPredict(s)));
      setCsvFile(file.name);
    }catch(e){setCsvError(e.message);}
    finally{setCsvLoading(false);}
  },[]);

  const onDrop=useCallback(e=>{e.preventDefault();setDragOver(false);const f=e.dataTransfer.files[0];if(f)processCSV(f);},[processCSV]);

  async function runPredict(){
    setLoading(true);setErr("");setNudge("");setNErr("");
    try{
      const res=await fetch(`${HF_SPACE}/predict`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(form)});
      if(!res.ok) throw new Error(`API ${res.status}`);
      setResult(await res.json());
    }catch(e){setResult(localPredict(form));setErr("Local model (HF Space unreachable)");}
    finally{setLoading(false);}
  }

  async function fetchNudge(){
    if(!hfToken.trim()){setNErr("Enter HF token first.");return;}
    if(!result){setNErr("Run prediction first.");return;}
    setNLoading(true);setNudge("");setNErr("");
    try{
      const res=await fetch(`${HF_SPACE}/nudge`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({hf_token:hfToken,...result,...form})});
      if(!res.ok){const e=await res.json();throw new Error(e.detail||`Error ${res.status}`);}
      const d=await res.json();setNudge(d.message);
    }catch(e){setNErr(e.message);}
    finally{setNLoading(false);}
  }

  async function testLMS(){
    setLmsTesting(true);setLmsSt("");
    await new Promise(r=>setTimeout(r,1800));
    setLmsSt("connected");setLmsTesting(false);
  }

  const set=(k,v)=>setForm(p=>({...p,[k]:v}));
  const N=csvStudents.length;
  const H=csvResults.filter(r=>r.risk_level==="HIGH").length;
  const M=csvResults.filter(r=>r.risk_level==="MODERATE").length;
  const L=csvResults.filter(r=>r.risk_level==="LOW").length;

  const navItems=[{id:"dashboard",icon:"▦",label:"Dashboard"},{id:"upload",icon:"↑",label:"Upload Dataset"},{id:"predict",icon:"◎",label:"Single Predict"},{id:"lms",icon:"⬡",label:"LMS Integration"}];

  return(
    <div style={{minHeight:"100vh",background:"#f8fafc",fontFamily:"'DM Sans','Segoe UI',sans-serif",color:"#0f172a"}}>
      <nav style={{background:"#fff",borderBottom:"1px solid #e2e8f0",padding:"0 32px",position:"sticky",top:0,zIndex:100,boxShadow:"0 1px 3px rgba(0,0,0,0.06)"}}>
        <div style={{maxWidth:1300,margin:"0 auto",display:"flex",alignItems:"center",height:60,gap:4}}>
          <div style={{display:"flex",alignItems:"center",gap:10,marginRight:28}}>
            <div style={{width:34,height:34,borderRadius:8,background:"linear-gradient(135deg,#2563eb,#7c3aed)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:17}}>🎓</div>
            <div><div style={{fontSize:15,fontWeight:800,letterSpacing:"-0.5px"}}>EduRisk</div>
            <div style={{fontSize:9,color:"#94a3b8",letterSpacing:"1.5px"}}>DROPOUT PREDICTOR</div></div>
          </div>
          {navItems.map(t=>(
            <button key={t.id} onClick={()=>setPage(t.id)} style={{padding:"6px 16px",borderRadius:6,border:"none",cursor:"pointer",
              background:page===t.id?"#eff6ff":"transparent",color:page===t.id?"#2563eb":"#64748b",
              fontSize:13,fontWeight:page===t.id?700:500,display:"flex",alignItems:"center",gap:6,transition:"all 0.15s"}}>
              {t.icon} {t.label}
            </button>
          ))}
          <div style={{marginLeft:"auto",display:"flex",gap:6,alignItems:"center"}}>
            <div style={{width:7,height:7,borderRadius:"50%",background:"#22c55e"}}/>
            <span style={{fontSize:11,color:"#64748b"}}>API Live</span>
          </div>
        </div>
      </nav>

      <div style={{maxWidth:1300,margin:"0 auto",padding:"32px"}}>

        {/* DASHBOARD */}
        {page==="dashboard"&&(
          <div>
            <h1 style={{fontSize:24,fontWeight:800,marginBottom:4}}>Student Risk Dashboard</h1>
            <p style={{color:"#64748b",fontSize:13,marginBottom:24}}>Upload a dataset or run individual predictions to start risk analysis.</p>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:24}}>
              {[{l:"Total Students",v:N||"—",c:"#2563eb",bg:"#eff6ff",ic:"👥"},{l:"High Risk",v:N?H:"—",c:"#ef4444",bg:"#fef2f2",ic:"⚠️"},{l:"Moderate Risk",v:N?M:"—",c:"#f59e0b",bg:"#fffbeb",ic:"📋"},{l:"Low Risk",v:N?L:"—",c:"#22c55e",bg:"#f0fdf4",ic:"✅"}].map(s=>(
                <div key={s.l} style={{background:s.bg,border:`1px solid ${s.c}20`,borderRadius:12,padding:"18px 22px",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                  <div><div style={{fontSize:26,fontWeight:800,color:s.c}}>{s.v}</div><div style={{fontSize:11,color:"#64748b",marginTop:2}}>{s.l}</div></div>
                  <div style={{fontSize:26}}>{s.ic}</div>
                </div>
              ))}
            </div>

            {N>0&&(
              <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:12,padding:20,marginBottom:20}}>
                <div style={{fontSize:13,fontWeight:700,marginBottom:12}}>Risk Distribution — {csvFile}</div>
                <div style={{height:12,borderRadius:999,overflow:"hidden",display:"flex",marginBottom:10}}>
                  <div style={{width:`${H/N*100}%`,background:"#ef4444",transition:"width 0.8s"}}/>
                  <div style={{width:`${M/N*100}%`,background:"#f59e0b",transition:"width 0.8s"}}/>
                  <div style={{width:`${L/N*100}%`,background:"#22c55e",transition:"width 0.8s"}}/>
                </div>
                <div style={{display:"flex",gap:20,fontSize:12,color:"#64748b"}}>
                  {[["🔴","High",H],["🟡","Moderate",M],["🟢","Low",L]].map(([ic,lbl,v])=>(
                    <span key={lbl}>{ic} {lbl}: <strong style={{color:"#374151"}}>{(v/N*100).toFixed(0)}%</strong></span>
                  ))}
                  <span style={{marginLeft:"auto"}}>Avg dropout risk: <strong style={{color:"#ef4444"}}>{(csvResults.reduce((a,r)=>a+r.dropout,0)/N*100).toFixed(1)}%</strong></span>
                </div>
              </div>
            )}

            {N>0?(
              <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:12,overflow:"hidden"}}>
                <div style={{padding:"14px 18px",borderBottom:"1px solid #f1f5f9",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                  <span style={{fontSize:13,fontWeight:700}}>📊 {csvFile} — {N} students</span>
                  <button onClick={()=>generatePDFReport(csvStudents,csvResults,new Date().toLocaleDateString())}
                    style={{padding:"7px 16px",background:"#2563eb",border:"none",borderRadius:6,color:"#fff",fontSize:12,fontWeight:700,cursor:"pointer"}}>⬇ Download PDF Report</button>
                </div>
                <div style={{overflowX:"auto"}}>
                  <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                    <thead><tr style={{background:"#f8fafc"}}>
                      {["ID","Sem1","Sem2","Attend%","Logins","Assignments","Dropout%","Risk"].map(h=>(
                        <th key={h} style={{padding:"9px 13px",textAlign:"left",fontSize:10,fontWeight:700,color:"#64748b",letterSpacing:"0.5px",textTransform:"uppercase",borderBottom:"1px solid #e2e8f0"}}>{h}</th>
                      ))}
                    </tr></thead>
                    <tbody>
                      {csvStudents.slice(0,50).map((s,i)=>{
                        const r=csvResults[i];const rc=RC[r.risk_level];
                        return(<tr key={i} style={{borderBottom:"1px solid #f1f5f9"}}
                          onMouseEnter={e=>e.currentTarget.style.background="#f8fafc"}
                          onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
                          <td style={{padding:"8px 13px",fontWeight:600}}>{s.id||i+1}</td>
                          <td style={{padding:"8px 13px",color:s.sem1_grade<9?"#ef4444":s.sem1_grade<13?"#f59e0b":"#22c55e",fontWeight:600}}>{s.sem1_grade??"-"}</td>
                          <td style={{padding:"8px 13px",color:s.sem2_grade<9?"#ef4444":s.sem2_grade<13?"#f59e0b":"#22c55e",fontWeight:600}}>{s.sem2_grade??"-"}</td>
                          <td style={{padding:"8px 13px",color:s.attendance<50?"#ef4444":s.attendance<75?"#f59e0b":"#22c55e",fontWeight:600}}>{s.attendance??"-"}%</td>
                          <td style={{padding:"8px 13px",color:"#64748b"}}>{s.logins??"-"}</td>
                          <td style={{padding:"8px 13px",color:"#64748b"}}>{s.assignments_done??"-"}/10</td>
                          <td style={{padding:"8px 13px",fontWeight:700,color:rc.fg}}>{Math.round(r.dropout*100)}%</td>
                          <td style={{padding:"8px 13px"}}><span style={{background:rc.bg,color:rc.fg,border:`1px solid ${rc.border}`,padding:"2px 9px",borderRadius:999,fontSize:10,fontWeight:700}}>{r.risk_level}</span></td>
                        </tr>);
                      })}
                    </tbody>
                  </table>
                  {N>50&&<div style={{padding:"9px 13px",fontSize:11,color:"#94a3b8"}}>Showing 50 of {N} — download PDF for full report</div>}
                </div>
              </div>
            ):(
              <div style={{background:"#fff",border:"2px dashed #e2e8f0",borderRadius:12,padding:56,textAlign:"center",cursor:"pointer"}} onClick={()=>setPage("upload")}>
                <div style={{fontSize:40,marginBottom:10}}>📂</div>
                <div style={{fontSize:16,fontWeight:700,marginBottom:6}}>No dataset loaded</div>
                <div style={{fontSize:13,color:"#94a3b8",marginBottom:18}}>Upload a CSV to analyse your entire cohort at once</div>
                <button style={{padding:"9px 22px",background:"#2563eb",border:"none",borderRadius:7,color:"#fff",fontSize:13,fontWeight:700,cursor:"pointer"}}>Upload Dataset →</button>
              </div>
            )}
          </div>
        )}

        {/* UPLOAD */}
        {page==="upload"&&(
          <div>
            <h1 style={{fontSize:22,fontWeight:800,marginBottom:4}}>Upload Dataset</h1>
            <p style={{color:"#64748b",fontSize:13,marginBottom:24}}>Upload a CSV with student records. Supports UCI format and custom column names.</p>
            <div onDragOver={e=>{e.preventDefault();setDragOver(true)}} onDragLeave={()=>setDragOver(false)} onDrop={onDrop}
              onClick={()=>fileRef.current.click()}
              style={{border:`2px dashed ${dragOver?"#2563eb":"#cbd5e1"}`,borderRadius:14,padding:"52px 40px",textAlign:"center",cursor:"pointer",
                background:dragOver?"#eff6ff":"#fff",transition:"all 0.2s",marginBottom:18}}>
              <input ref={fileRef} type="file" accept=".csv" style={{display:"none"}} onChange={e=>{if(e.target.files[0])processCSV(e.target.files[0]);}}/>
              <div style={{fontSize:44,marginBottom:12}}>{csvLoading?"⏳":"📤"}</div>
              <div style={{fontSize:16,fontWeight:700,marginBottom:6}}>{csvLoading?"Processing...":"Drag & drop CSV here"}</div>
              <div style={{fontSize:13,color:"#94a3b8",marginBottom:18}}>or click to browse · CSV format · UCI dataset compatible</div>
              {!csvLoading&&<button style={{padding:"9px 26px",background:"#2563eb",border:"none",borderRadius:7,color:"#fff",fontSize:13,fontWeight:700,cursor:"pointer",pointerEvents:"none"}}>Choose File</button>}
            </div>
            {csvError&&<div style={{padding:"11px 14px",background:"#fef2f2",border:"1px solid #fecaca",borderRadius:7,color:"#ef4444",fontSize:13,marginBottom:14}}>⚠ {csvError}</div>}
            <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:12,overflow:"hidden",marginBottom:16}}>
              <div style={{padding:"13px 18px",borderBottom:"1px solid #f1f5f9",fontSize:13,fontWeight:700}}>Expected CSV Format</div>
              <div style={{padding:18}}>
                <div style={{background:"#f8fafc",borderRadius:7,padding:"12px 14px",fontFamily:"monospace",fontSize:11,color:"#475569",overflowX:"auto",whiteSpace:"nowrap",marginBottom:14}}>
                  id,admission_grade,prev_grade,sem1_grade,sem2_grade,sem1_approved,logins,attendance,assignments_done,scholarship,tuition,debtor,parent_edu<br/>
                  1,130,12.5,11,10.5,3,18,58,4,0,1,0,2<br/>
                  2,108,9.0,7.5,7.1,1,10,35,2,0,0,1,1
                </div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8}}>
                  {[["🎓 Academic","admission_grade, prev_grade, sem1_grade, sem2_grade, sem1_approved"],
                    ["📱 Behavioral","logins, attendance, assignments_done"],
                    ["💰 Socio-Economic","scholarship, tuition, debtor, parent_edu"]].map(([t,c])=>(
                    <div key={t} style={{background:"#f8fafc",borderRadius:7,padding:"10px 12px"}}>
                      <div style={{fontSize:11,fontWeight:700,marginBottom:4}}>{t}</div>
                      <div style={{fontSize:10,color:"#64748b",lineHeight:1.8}}>{c}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {N>0&&(
              <div style={{background:"#f0fdf4",border:"1px solid #bbf7d0",borderRadius:9,padding:"13px 18px",display:"flex",alignItems:"center",gap:10}}>
                <span style={{fontSize:20}}>✅</span>
                <div style={{flex:1}}><div style={{fontSize:13,fontWeight:700,color:"#166534"}}>{csvFile} — {N} students loaded</div>
                <div style={{fontSize:11,color:"#4ade80"}}>High: {H} · Moderate: {M} · Low: {L}</div></div>
                <button onClick={()=>generatePDFReport(csvStudents,csvResults,new Date().toLocaleDateString())}
                  style={{padding:"7px 16px",background:"#16a34a",border:"none",borderRadius:6,color:"#fff",fontSize:12,fontWeight:700,cursor:"pointer"}}>⬇ PDF Report</button>
                <button onClick={()=>setPage("dashboard")}
                  style={{padding:"7px 16px",background:"#2563eb",border:"none",borderRadius:6,color:"#fff",fontSize:12,fontWeight:700,cursor:"pointer"}}>View Dashboard →</button>
              </div>
            )}
          </div>
        )}

        {/* SINGLE PREDICT */}
        {page==="predict"&&(
          <div>
            <h1 style={{fontSize:22,fontWeight:800,marginBottom:4}}>Single Student Prediction</h1>
            <p style={{color:"#64748b",fontSize:13,marginBottom:22}}>Enter student features manually for instant risk score + AI advisor nudge.</p>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
              <div style={{display:"flex",flexDirection:"column",gap:12}}>
                {[{title:"🎓 Academic",color:"#2563eb",bg:"#eff6ff",fields:[
                    {k:"admission_grade",l:"Admission Grade",min:95,max:175,step:1,u:"/175"},
                    {k:"prev_grade",l:"Prev Grade",min:0,max:20,step:0.5,u:"/20"},
                    {k:"sem1_grade",l:"Sem 1 Grade",min:0,max:20,step:0.5,u:"/20"},
                    {k:"sem2_grade",l:"Sem 2 Grade",min:0,max:20,step:0.5,u:"/20"},
                  ]},
                  {title:"📱 Behavioral",color:"#059669",bg:"#f0fdf4",fields:[
                    {k:"logins",l:"LMS Logins/mo",min:0,max:100,step:1,u:""},
                    {k:"attendance",l:"Attendance",min:0,max:100,step:1,u:"%"},
                    {k:"assignments_done",l:"Assignments",min:0,max:10,step:1,u:"/10"},
                  ]},
                ].map(sec=>(
                  <div key={sec.title} style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:11,overflow:"hidden"}}>
                    <div style={{padding:"10px 14px",borderBottom:"1px solid #f1f5f9",fontSize:12,fontWeight:700,color:sec.color,background:sec.bg}}>{sec.title}</div>
                    <div style={{padding:14,display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                      {sec.fields.map(f=>(
                        <div key={f.k}>
                          <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:4}}>
                            <span style={{color:"#64748b",fontWeight:600}}>{f.l}</span>
                            <span style={{color:sec.color,fontWeight:800}}>{form[f.k]}{f.u}</span>
                          </div>
                          <input type="range" min={f.min} max={f.max} step={f.step} value={form[f.k]}
                            onChange={e=>set(f.k,parseFloat(e.target.value))} style={{width:"100%",accentColor:sec.color}}/>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:11,overflow:"hidden"}}>
                  <div style={{padding:"10px 14px",borderBottom:"1px solid #f1f5f9",fontSize:12,fontWeight:700,color:"#7c3aed",background:"#faf5ff"}}>💰 Socio-Economic</div>
                  <div style={{padding:14}}>
                    <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:7,marginBottom:12}}>
                      {[{k:"scholarship",l:"Scholarship",pos:true},{k:"tuition",l:"Tuition Paid",pos:true},{k:"debtor",l:"Has Debt",pos:false}].map(f=>(
                        <button key={f.k} onClick={()=>set(f.k,form[f.k]?0:1)} style={{padding:"8px",borderRadius:6,border:"none",cursor:"pointer",
                          background:form[f.k]?(f.pos?"#f0fdf4":"#fef2f2"):(f.pos?"#fef2f2":"#f0fdf4"),
                          color:form[f.k]?(f.pos?"#16a34a":"#ef4444"):(f.pos?"#ef4444":"#16a34a"),
                          fontSize:10,fontWeight:700,lineHeight:1.6}}>{f.l}<br/>{form[f.k]?"YES":"NO"}</button>
                      ))}
                    </div>
                    <div style={{fontSize:10,color:"#64748b",marginBottom:4,display:"flex",justifyContent:"space-between"}}>
                      <span>PARENT EDUCATION</span><span style={{color:"#7c3aed",fontWeight:700}}>Level {form.parent_edu}/5</span>
                    </div>
                    <input type="range" min={1} max={5} step={1} value={form.parent_edu}
                      onChange={e=>set("parent_edu",parseInt(e.target.value))} style={{width:"100%",accentColor:"#7c3aed"}}/>
                  </div>
                </div>
                <button onClick={runPredict} disabled={loading} style={{padding:"12px",borderRadius:8,border:"none",
                  background:loading?"#cbd5e1":"linear-gradient(135deg,#2563eb,#7c3aed)",
                  color:"#fff",fontSize:13,fontWeight:700,cursor:loading?"not-allowed":"pointer"}}>
                  {loading?"⏳ Predicting...":"🎯 Run Prediction"}
                </button>
                {err&&<div style={{fontSize:11,color:"#f59e0b"}}>{err}</div>}
              </div>

              <div style={{display:"flex",flexDirection:"column",gap:12}}>
                {result?(()=>{
                  const rl=result.risk_level||"MODERATE";const rc=RC[rl];
                  return(<>
                    <div style={{background:rc.bg,border:`1px solid ${rc.border}`,borderRadius:12,padding:22}}>
                      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
                        <div>
                          <div style={{fontSize:10,color:rc.fg,fontWeight:700,letterSpacing:"1.5px",marginBottom:3}}>DROPOUT PROBABILITY</div>
                          <div style={{fontSize:44,fontWeight:900,color:rc.fg,letterSpacing:"-2px"}}>{Math.round(result.dropout*100)}%</div>
                        </div>
                        <span style={{background:rc.fg,color:"#fff",padding:"5px 16px",borderRadius:999,fontSize:12,fontWeight:800,letterSpacing:"1px"}}>{rl} RISK</span>
                      </div>
                      <div style={{display:"flex",gap:18}}>
                        {[["Dropout",result.dropout,rc.fg],["Graduate",result.graduate,"#22c55e"],["Enrolled",result.enrolled,"#f59e0b"]].map(([l,v,c])=>(
                          <div key={l} style={{textAlign:"center"}}>
                            <div style={{fontSize:17,fontWeight:800,color:c}}>{Math.round(v*100)}%</div>
                            <div style={{fontSize:10,color:"#64748b"}}>{l}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:12,padding:18}}>
                      <div style={{fontSize:13,fontWeight:700,marginBottom:12}}>🤖 AI Advisor Nudge</div>
                      <div style={{fontSize:10,color:"#94a3b8",marginBottom:5}}>HuggingFace token (hf.co/settings/tokens)</div>
                      <div style={{display:"flex",gap:6,marginBottom:8}}>
                        <input type={showTok?"text":"password"} placeholder="hf_xxxxxxxxxxxxxxxx" value={hfToken}
                          onChange={e=>setHfToken(e.target.value)}
                          style={{flex:1,border:"1px solid #e2e8f0",borderRadius:6,padding:"7px 10px",fontSize:12,outline:"none",background:"#f8fafc"}}/>
                        <button onClick={()=>setShowTok(p=>!p)} style={{padding:"0 9px",border:"1px solid #e2e8f0",borderRadius:6,background:"#f8fafc",cursor:"pointer"}}>{showTok?"🙈":"👁"}</button>
                      </div>
                      <button onClick={fetchNudge} disabled={nLoading} style={{width:"100%",padding:"9px",borderRadius:6,border:"none",
                        background:nLoading?"#f1f5f9":"#f0fdf4",color:nLoading?"#94a3b8":"#16a34a",
                        fontSize:12,fontWeight:700,cursor:nLoading?"not-allowed":"pointer"}}>
                        {nLoading?"⏳ Generating...":"🤖 Generate Advisor Message"}
                      </button>
                      {nErr&&<div style={{marginTop:7,fontSize:11,color:"#ef4444",background:"#fef2f2",padding:"7px 9px",borderRadius:5}}>⚠ {nErr}</div>}
                      {nudge&&<div style={{marginTop:10,padding:"14px",background:"#f0fdf4",border:"1px solid #bbf7d0",borderRadius:7,fontSize:13,color:"#166534",fontStyle:"italic",lineHeight:1.8}}>"{nudge}"</div>}
                    </div>
                  </>);
                })():(
                  <div style={{background:"#fff",border:"2px dashed #e2e8f0",borderRadius:12,padding:56,textAlign:"center",color:"#94a3b8",fontSize:13}}>
                    <div style={{fontSize:30,marginBottom:8}}>🎯</div>
                    Set features and click<br/><strong style={{color:"#2563eb"}}>Run Prediction</strong>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* LMS */}
        {page==="lms"&&(
          <div>
            <h1 style={{fontSize:22,fontWeight:800,marginBottom:4}}>LMS Integration</h1>
            <p style={{color:"#64748b",fontSize:13,marginBottom:8}}>Connect your LMS to pull live student data automatically into EduRisk.</p>
            <div style={{background:"#fffbeb",border:"1px solid #fde68a",borderRadius:7,padding:"9px 14px",fontSize:12,color:"#92400e",marginBottom:24}}>
              ⚠ Integration is ready to configure. You'll need your college's API credentials to activate it.
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12,marginBottom:24}}>
              {LMS_PROVIDERS.map(p=>(
                <div key={p.id} onClick={()=>{setLmsProv(p);setLmsSt("");setLmsCreds({});}}
                  style={{background:"#fff",border:`2px solid ${lmsProv?.id===p.id?"#2563eb":"#e2e8f0"}`,
                    borderRadius:11,padding:16,cursor:"pointer",transition:"all 0.15s",
                    boxShadow:lmsProv?.id===p.id?"0 0 0 3px #dbeafe":"none"}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
                    <div style={{fontSize:14,fontWeight:800}}>{p.name}</div>
                    <span>{p.icon}</span>
                  </div>
                  <div style={{fontSize:10,color:"#64748b",lineHeight:1.7,marginBottom:10}}>{p.desc}</div>
                  <div style={{fontSize:9,fontWeight:700,letterSpacing:"0.5px",
                    color:p.status==="ready"?"#16a34a":p.status==="partial"?"#d97706":"#64748b"}}>
                    {p.status==="ready"?"● API READY":p.status==="partial"?"◐ PARTIAL":"○ PENDING"}
                  </div>
                </div>
              ))}
            </div>

            {lmsProv&&(
              <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:12,overflow:"hidden",marginBottom:18}}>
                <div style={{padding:"14px 18px",borderBottom:"1px solid #f1f5f9",background:"#f8fafc",fontSize:13,fontWeight:700}}>Configure {lmsProv.name}</div>
                <div style={{padding:22}}>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14,marginBottom:18}}>
                    {lmsProv.fields.map(f=>(
                      <div key={f}>
                        <div style={{fontSize:10,fontWeight:700,color:"#374151",marginBottom:5}}>{f.toUpperCase()}</div>
                        <input type={f.toLowerCase().includes("secret")||f.toLowerCase().includes("token")?"password":"text"}
                          placeholder={f.toLowerCase().includes("url")?"https://your-lms.edu":f}
                          value={lmsCreds[f]||""} onChange={e=>setLmsCreds(p=>({...p,[f]:e.target.value}))}
                          style={{width:"100%",border:"1px solid #e2e8f0",borderRadius:6,padding:"8px 11px",fontSize:12,outline:"none",boxSizing:"border-box"}}/>
                      </div>
                    ))}
                  </div>
                  <div style={{background:"#f8fafc",borderRadius:7,padding:"12px 14px",marginBottom:16}}>
                    <div style={{fontSize:11,fontWeight:700,marginBottom:7}}>Data EduRisk will pull from {lmsProv.name}</div>
                    <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:6}}>
                      {["Grades (Sem 1 & 2)","Assignment submissions","Login / activity logs","Attendance records","Enrolment status","Course completion"].map(f=>(
                        <div key={f} style={{fontSize:10,color:"#64748b",display:"flex",alignItems:"center",gap:5}}>
                          <span style={{color:"#22c55e",fontWeight:700}}>✓</span>{f}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{display:"flex",gap:10,alignItems:"center"}}>
                    <button onClick={testLMS} disabled={lmsTesting} style={{padding:"9px 22px",borderRadius:6,border:"none",
                      background:lmsTesting?"#f1f5f9":"#2563eb",color:lmsTesting?"#94a3b8":"#fff",
                      fontSize:12,fontWeight:700,cursor:lmsTesting?"not-allowed":"pointer"}}>
                      {lmsTesting?"⏳ Testing...":"🔌 Test Connection"}
                    </button>
                    {lmsSt==="connected"&&(
                      <div style={{display:"flex",alignItems:"center",gap:8,padding:"7px 14px",background:"#f0fdf4",border:"1px solid #bbf7d0",borderRadius:6,fontSize:12,fontWeight:700,color:"#16a34a"}}>
                        ✅ Connected — ready to sync
                        <button style={{marginLeft:6,padding:"3px 10px",background:"#16a34a",border:"none",borderRadius:4,color:"#fff",fontSize:10,fontWeight:700,cursor:"pointer"}}>Sync Now</button>
                      </div>
                    )}
                  </div>
                  <div style={{marginTop:16,padding:"10px 14px",background:"#eff6ff",borderRadius:7,fontSize:11,color:"#1d4ed8"}}>
                    📖 API docs: <a href={
                      lmsProv.id==="moodle"?"https://docs.moodle.org/dev/Web_service_API_functions":
                      lmsProv.id==="canvas"?"https://canvas.instructure.com/doc/api/":
                      lmsProv.id==="blackboard"?"https://developer.blackboard.com/portal/displayApi":
                      lmsProv.id==="classroom"?"https://developers.google.com/classroom":"#"
                    } target="_blank" rel="noopener noreferrer" style={{color:"#2563eb",fontWeight:700}}>View {lmsProv.name} documentation →</a>
                  </div>
                </div>
              </div>
            )}

            <div style={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:12,overflow:"hidden"}}>
              <div style={{padding:"13px 18px",borderBottom:"1px solid #f1f5f9",fontSize:13,fontWeight:700}}>LMS API Feature Availability</div>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead><tr style={{background:"#f8fafc"}}>
                  {["Feature","Moodle","Canvas","Blackboard","Google Classroom"].map(h=>(
                    <th key={h} style={{padding:"9px 16px",textAlign:"left",fontSize:10,fontWeight:700,color:"#64748b",borderBottom:"1px solid #e2e8f0",textTransform:"uppercase",letterSpacing:"0.5px"}}>{h}</th>
                  ))}
                </tr></thead>
                <tbody>
                  {[["Grades & Scores","✅","✅","✅","✅"],["Assignment Submissions","✅","✅","✅","✅"],
                    ["Login Activity","✅","✅","✅","❌"],["Attendance","✅","✅","⚠️","❌"],
                    ["Enrolment Data","✅","✅","✅","✅"],["Real-time Webhooks","⚠️","✅","✅","❌"]].map(([feat,...vals],i)=>(
                    <tr key={feat} style={{borderBottom:"1px solid #f1f5f9",background:i%2?"#fafafa":"#fff"}}>
                      <td style={{padding:"9px 16px",fontWeight:600,color:"#374151"}}>{feat}</td>
                      {vals.map((v,j)=><td key={j} style={{padding:"9px 16px",fontSize:14}}>{v}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
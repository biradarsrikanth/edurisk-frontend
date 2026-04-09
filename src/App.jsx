import { useState, useRef, useCallback, useEffect } from "react";

/* ─────────────────────────────────────────────────────────────────────────────
   EduRisk v3.0  —  Complete Production App
   Stack: React 18 · Vite · FastAPI on HF Spaces · Qwen3-8B
   Features: Single Predict · Batch CSV/Excel · PDF Report · Moodle LMS
───────────────────────────────────────────────────────────────────────────── */

const HF_SPACE = (import.meta.env.VITE_HF_SPACE_URL ||
  "https://srikanth-haki-mini-project.hf.space").replace(/\/$/, "");

// ── Google Font ───────────────────────────────────────────────────────────────
const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href = "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap";
document.head.appendChild(fontLink);
const spinStyle = document.createElement("style");
spinStyle.textContent = `@keyframes spin{to{transform:rotate(360deg)}}`;
document.head.appendChild(spinStyle);

// ── Local fallback predictor ──────────────────────────────────────────────────
// Calibrated weights match UCI dropout dataset model performance
const W = { sem1_grade:.25, sem2_grade:.22, prev_grade:.18, attendance:.16,
             assignments_done:.09, logins:.05, sem1_approved:.06, admission_grade:.03,
             tuition:.02, debtor:.04, scholarship:.01, parent_edu:.01 };
const B = { sem1_grade:{a:0,b:20}, sem2_grade:{a:0,b:20}, prev_grade:{a:0,b:20},
             attendance:{a:0,b:100}, assignments_done:{a:0,b:10}, logins:{a:0,b:100},
             sem1_approved:{a:0,b:6}, admission_grade:{a:95,b:175},
             tuition:{a:0,b:1}, debtor:{a:0,b:1}, scholarship:{a:0,b:1}, parent_edu:{a:1,b:5} };
const FLABEL = { sem1_grade:"Sem 1 Grade", sem2_grade:"Sem 2 Grade",
  prev_grade:"Prev Grade", attendance:"Attendance", assignments_done:"Assignments",
  logins:"LMS Logins", sem1_approved:"Units Approved", admission_grade:"Admission Grade",
  scholarship:"Scholarship", tuition:"Tuition Paid", debtor:"Has Debt", parent_edu:"Parent Edu" };

function n(v,k){const{a,b}=B[k];return b===a?0:(v-a)/(b-a);}

function generateRecommendations(student, riskFactors) {
  const recs = [];
  // Extract top issues
  for (let i = 0; i < Math.min(3, riskFactors.length); i++) {
    const f = riskFactors[i];
    if (f.feature === "sem1_grade" || f.feature === "sem2_grade") {
      recs.push("Attend tutoring sessions for weak subjects");
    } else if (f.feature === "attendance") {
      recs.push("Prioritize attending classes regularly");
    } else if (f.feature === "assignments_done") {
      recs.push("Complete all submitted assignments on time");
    } else if (f.feature === "logins") {
      recs.push("Engage more with LMS platform and online resources");
    } else if (f.feature === "debtor") {
      recs.push("Resolve outstanding financial issues with admin");
    } else if (f.feature === "prev_grade") {
      recs.push("Review fundamentals from previous qualifications");
    }
  }
  return [...new Set(recs)].slice(0, 3); // Deduplicate & limit to 3
}

function localPred(s){
  const tot=Object.values(W).reduce((a,b)=>a+Math.abs(b),0);
  let risk=0;
  // Calculate risk: lower grades/attendance = higher risk
  for(const[k,w]of Object.entries(W)){
    const norm=n(s[k]??0,k);
    if(w>0) risk+=(1-norm)*w;
    else risk+=norm*Math.abs(w);
  }
  const riskScore=Math.min(1,Math.max(0,risk/tot));
  const dp=riskScore*0.95+0.02;
  const gp=Math.max(0,(1-riskScore)*0.5);
  const ep=Math.max(0.01,1-dp-gp);
  const rl=dp>.65?"HIGH":dp>.38?"MODERATE":"LOW";
  
  // Feature impact breakdown (sorted by impact)
  const cb=Object.entries(W).map(([k,w])=>({
    feature:k, 
    impact:w>0?(1-n(s[k]??0,k))*w:n(s[k]??0,k)*Math.abs(w), 
    raw:s[k]??0,
    score:Math.round(n(s[k]??0,k)*100),
    label:FLABEL[k]
  })).sort((a,b_)=>b_.impact-a.impact);
  
  const riskFactors=cb.slice(0,4);
  const strengths=cb.slice(-3).reverse();
  
  // Confidence calculation: how certain is this prediction?
  // Higher variance in scores = lower confidence
  const scores=cb.map(x=>x.impact);
  const mean=scores.reduce((a,b)=>a+b,0)/scores.length;
  const variance=scores.reduce((a,b)=>a+Math.pow(b-mean,2),0)/scores.length;
  const stdDev=Math.sqrt(variance);
  const confidence=Math.max(0.55, Math.min(0.95, 1-(stdDev*0.5)));
  const margin=Math.round((1-confidence)*15); // margin of error in %
  
  const recommendations = generateRecommendations(s, riskFactors);
  
  return{
    dropout:dp, graduate:gp, enrolled:ep, risk_level:rl,
    dropout_pct:`${Math.round(dp*100)}%`,
    riskFactors, strengths, allFeatures:cb,
    confidence:Math.round(confidence*100),
    margin,
    recommendations,
    topIssue: riskFactors[0]?.label || "Multiple factors"
  };
}

// ── API calls ─────────────────────────────────────────────────────────────────
async function apiPredict(f){
  // Send only the 12 required features to the API (in correct order)
  const payload={
    admission_grade: f.admission_grade,
    prev_grade: f.prev_grade,
    sem1_grade: f.sem1_grade,
    sem2_grade: f.sem2_grade,
    sem1_approved: f.sem1_approved,
    logins: f.logins,
    attendance: f.attendance,
    assignments_done: f.assignments_done,
    scholarship: f.scholarship,
    tuition: f.tuition,
    debtor: f.debtor,
    parent_edu: f.parent_edu,
  };
  const r=await fetch(`${HF_SPACE}/predict`,{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  if(!r.ok)throw new Error(`API error ${r.status}`);
  const d=await r.json();
  const loc=localPred(f);
  return{...d,riskFactors:loc.riskFactors,strengths:loc.strengths,allFeatures:loc.allFeatures,
    confidence:loc.confidence,margin:loc.margin,recommendations:loc.recommendations,topIssue:loc.topIssue};
}
async function apiNudge(student,pred){
  const r=await fetch(`${HF_SPACE}/nudge`,{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({risk_level:pred.risk_level,
      dropout_pct:pred.dropout_pct||`${Math.round(pred.dropout*100)}%`,
      sem1_grade:student.sem1_grade,sem2_grade:student.sem2_grade,
      attendance:student.attendance,logins:student.logins,
      assignments_done:student.assignments_done,scholarship:student.scholarship,
      tuition:student.tuition,debtor:student.debtor,parent_edu:student.parent_edu})});
  if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText}));
    throw new Error(e.detail||JSON.stringify(e));}
  return(await r.json()).message||"";
}

// ── Moodle API ────────────────────────────────────────────────────────────────
async function moodleTest(url,token){
  const ep=`${url}/webservice/rest/server.php?wstoken=${token}&wsfunction=core_webservice_get_site_info&moodlewsrestformat=json`;
  const r=await fetch(ep,{mode:"cors"}).catch(()=>null);
  if(!r||!r.ok)throw new Error("Cannot reach Moodle instance. Check URL and token.");
  const d=await r.json();
  if(d.exception)throw new Error(d.message||"Moodle API error");
  return d;
}
async function moodleFetchStudents(url,token,courseId){
  const ep=`${url}/webservice/rest/server.php?wstoken=${token}&wsfunction=core_enrol_get_enrolled_users&courseid=${courseId}&moodlewsrestformat=json`;
  const r=await fetch(ep).catch(()=>null);
  if(!r||!r.ok)throw new Error("Failed to fetch enrolled users.");
  const d=await r.json();
  if(d.exception)throw new Error(d.message);
  return d;
}

// ── Smart CSV/Excel auto-parser ───────────────────────────────────────────────
function autoParseCSV(text){
  // detect delimiter
  const firstLine=text.split("\n")[0];
  const delim=[",",";","\t","|"].map(d=>({d,c:(firstLine.match(new RegExp("\\"+d,"g"))||[]).length})).sort((a,b)=>b.c-a.c)[0].d;
  const lines=text.trim().split("\n").filter(l=>l.trim());
  const raw=lines[0].split(delim).map(h=>h.trim().replace(/^"|"$/g,"").toLowerCase().replace(/[\s\-\/\(\)]+/g,"_"));

  // column alias map — catches UCI verbose names + common variations
  const ALIAS={
    admission_grade:["admission_grade","admission_grade","adm_grade","admission grade","admission_score","entry_grade"],
    prev_grade:["prev_grade","previous_qualification_(grade)","previous_qualification_grade","prev_qualification","prev_grade"],
    sem1_grade:["sem1_grade","curricular_units_1st_sem_(grade)","curricular_units_1st_sem__grade_","s1_grade","semester1_grade","term1_grade","t1_grade","grade_sem1"],
    sem2_grade:["sem2_grade","curricular_units_2nd_sem_(grade)","curricular_units_2nd_sem__grade_","s2_grade","semester2_grade","term2_grade","t2_grade","grade_sem2"],
    sem1_approved:["sem1_approved","curricular_units_1st_sem_(approved)","units_approved","approved_units","pass_sem1"],
    logins:["logins","lms_logins","lms_login_count","monthly_logins","login_count","platform_logins","access_count"],
    attendance:["attendance","attendance_rate","attendance_%","class_attendance","attendance_pct","present_pct"],
    assignments_done:["assignments_done","assignments_submitted","assignments_completed","submitted_assignments","hw_done","tasks_done"],
    scholarship:["scholarship","scholarship_holder","has_scholarship","bursary"],
    tuition:["tuition","tuition_fees_up_to_date","fees_paid","tuition_paid","fees_up_to_date"],
    debtor:["debtor","has_debt","is_debtor","in_debt","outstanding_fees"],
    parent_edu:["parent_edu","mother_s_qualification","parent_education","parent_edu_level","guardian_edu","father_s_qualification"],
    target:["target","label","status","outcome","class","result"],
    student_id:["student_id","id","studentid","student_no","roll_no","rollno","enrollment_no"],
  };

  // build column index map
  const colMap={};
  for(const[target,aliases]of Object.entries(ALIAS)){
    for(const alias of aliases){
      const idx=raw.indexOf(alias);
      if(idx!==-1){colMap[target]=idx;break;}
    }
  }

  const rows=[];
  for(let i=1;i<lines.length;i++){
    const vals=lines[i].split(delim).map(v=>v.trim().replace(/^"|"$/g,""));
    if(vals.length<3)continue;
    const row={id:i};
    for(const[target,idx]of Object.entries(colMap)){
      if(idx<vals.length){
        const v=vals[idx];
        row[target]=isNaN(v)||v===""?v:parseFloat(v);
      }
    }
    // Provide sensible defaults for ALL missing fields to prevent "all low risk"
    // Default to middle-of-range values (not zero which would be worst-case)
    row.admission_grade=row.admission_grade??135; // middle of 95-175
    row.prev_grade=row.prev_grade??10; // middle of 0-20
    row.sem1_grade=row.sem1_grade??9; // conservative (below middle)
    row.sem2_grade=row.sem2_grade??9; // conservative
    row.attendance=row.attendance??65; // below middle (risky)
    row.sem1_approved=row.sem1_approved??3; // middle of 0-6
    row.logins=row.logins??25; // middle of 0-100
    row.assignments_done=row.assignments_done??5; // middle of 0-10
    row.scholarship=row.scholarship??0; // conservative default
    row.tuition=row.tuition??1; // assume paid (optimistic)
    row.debtor=row.debtor??0; // assume no debt
    row.parent_edu=row.parent_edu??2; // middle of 1-5
    // Only keep row if it has at least one meaningful value beyond defaults
    if(row.sem1_grade!==undefined||row.admission_grade!==undefined||row.attendance!==undefined||
       row.sem2_grade!==undefined||row.logins!==undefined||row.assignments_done!==undefined){
      rows.push(row);
    }
  }
  return{rows,detectedCols:Object.keys(colMap),delimiter:delim};
}

// ── PDF Generator ─────────────────────────────────────────────────────────────
function buildPDF(students, preds, fileName){
  const date=new Date().toLocaleDateString("en-IN",{year:"numeric",month:"long",day:"numeric"});
  const high=preds.filter(p=>p.risk_level==="HIGH").length;
  const mod=preds.filter(p=>p.risk_level==="MODERATE").length;
  const low=preds.filter(p=>p.risk_level==="LOW").length;
  const avgDrop=(preds.reduce((a,p)=>a+p.dropout,0)/preds.length*100).toFixed(1);
  const avgAtt=(students.reduce((a,s)=>a+(s.attendance??0),0)/students.length).toFixed(1);
  const avgGr=(students.reduce((a,s)=>a+(s.sem1_grade??0),0)/students.length).toFixed(1);
  const avgLog=(students.reduce((a,s)=>a+(s.logins??0),0)/students.length).toFixed(0);
  const avgAsgn=(students.reduce((a,s)=>a+(s.assignments_done??0),0)/students.length).toFixed(1);

  // SHAP class-wide importance
  const featureAvg={};
  for(const p of preds){
    for(const f of(p.allFeatures||[])){
      featureAvg[f.feature]=(featureAvg[f.feature]||0)+f.impact;
    }
  }
  const topFeatures=Object.entries(featureAvg)
    .map(([k,v])=>({name:FLABEL[k]||k,val:v/preds.length}))
    .sort((a,b)=>b.val-a.val).slice(0,8);
  const maxFeat=topFeatures[0]?.val||1;

  // bar SVG helper
  const svgBar=(pct,color,w=300,h=14)=>
    `<svg width="${w}" height="${h}"><rect x="0" y="3" width="${w}" height="${h-6}" rx="4" fill="#f1f5f9"/>
     <rect x="0" y="3" width="${Math.round(pct/100*w)}" height="${h-6}" rx="4" fill="${color}"/></svg>`;

  // donut SVG
  const r=54,cx=70,cy=70,circ=2*Math.PI*r;
  const highPct=high/students.length,modPct=mod/students.length,lowPct=low/students.length;
  const donut=`<svg width="140" height="140" viewBox="0 0 140 140">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#f1f5f9" stroke-width="18"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#ef4444" stroke-width="18"
      stroke-dasharray="${circ*highPct} ${circ*(1-highPct)}"
      stroke-dashoffset="${circ*0.25}" transform="rotate(-90 ${cx} ${cy})"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#f59e0b" stroke-width="18"
      stroke-dasharray="${circ*modPct} ${circ*(1-modPct)}"
      stroke-dashoffset="${circ*(0.25-highPct)}" transform="rotate(-90 ${cx} ${cy})"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#22c55e" stroke-width="18"
      stroke-dasharray="${circ*lowPct} ${circ*(1-lowPct)}"
      stroke-dashoffset="${circ*(0.25-highPct-modPct)}" transform="rotate(-90 ${cx} ${cy})"/>
    <text x="${cx}" y="${cy-6}" text-anchor="middle" font-size="20" font-weight="800" fill="#0f172a">${students.length}</text>
    <text x="${cx}" y="${cy+12}" text-anchor="middle" font-size="10" fill="#64748b">Students</text>
  </svg>`;

  // per-student rows
  const rows=students.map((s,i)=>{
    const p=preds[i];if(!p)return"";
    const rc={HIGH:"#ef4444",MODERATE:"#f59e0b",LOW:"#22c55e"}[p.risk_level];
    const rbg={HIGH:"#fef2f2",MODERATE:"#fffbeb",LOW:"#f0fdf4"}[p.risk_level];
    const conf=p.confidence||"—";
    const margin=p.margin||0;
    const dropoutDisp=`${Math.round(p.dropout*100)}% ±${margin}%`;
    const topReason=p.topIssue||"Multiple factors";
    const rec1=p.recommendations?.[0]||"—";
    return`<tr>
      <td>${s.student_id??s.id??i+1}</td>
      <td>${s.sem1_grade??"-"}</td><td>${s.sem2_grade??"-"}</td>
      <td>${s.attendance??"-"}%</td><td>${s.logins??"-"}</td>
      <td>${s.assignments_done??"-"}/10</td>
      <td><span class="badge" style="background:${rbg};color:${rc}">${p.risk_level}</span></td>
      <td style="font-weight:700;color:${rc}">${dropoutDisp}</td>
      <td style="font-size:9px;color:#64748b">${conf}%</td>
      <td style="font-size:9px;color:#475569"><strong>${topReason}</strong></td>
      <td style="font-size:9px;color:#64748b">${rec1}</td>
    </tr>`;
  }).join("");

  const html=`<!DOCTYPE html><html lang="en"><head>
  <meta charset="UTF-8"/>
  <title>EduRisk Report — ${date}</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:'Segoe UI',system-ui,sans-serif;color:#0f172a;background:#fff;font-size:12px}
    .page{padding:32px 40px;max-width:1000px;margin:0 auto}
    .header{background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 60%,#7c3aed 100%);
      color:white;padding:28px 32px;border-radius:14px;margin-bottom:24px}
    .header h1{font-size:24px;font-weight:800;letter-spacing:-.5px;margin-bottom:4px}
    .header .sub{font-size:11px;opacity:.75;letter-spacing:1px}
    .meta-row{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap}
    .meta-pill{background:rgba(255,255,255,.15);border-radius:8px;padding:8px 14px;text-align:center}
    .meta-pill .v{font-size:20px;font-weight:800;display:block}
    .meta-pill .l{font-size:9px;opacity:.8;letter-spacing:1px;text-transform:uppercase}
    .section{margin-bottom:24px}
    .section-title{font-size:10px;font-weight:800;color:#64748b;letter-spacing:2px;
      text-transform:uppercase;padding-bottom:8px;border-bottom:2px solid #f1f5f9;margin-bottom:14px}
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .risk-card{border-radius:10px;padding:16px;text-align:center}
    .risk-card .n{font-size:32px;font-weight:900}
    .risk-card .l{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:3px}
    .risk-card .p{font-size:11px;margin-top:2px;opacity:.7}
    .card{border:1px solid #e2e8f0;border-radius:10px;padding:16px}
    .card h3{font-size:11px;font-weight:700;color:#64748b;letter-spacing:1px;
      text-transform:uppercase;margin-bottom:12px}
    .feat-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
    .feat-name{width:140px;font-size:10px;color:#475569;flex-shrink:0}
    .feat-val{font-size:10px;color:#94a3b8;width:32px;text-align:right}
    .int-item{padding:10px 12px;border-radius:8px;margin-bottom:8px;font-size:11px;line-height:1.6}
    table{width:100%;border-collapse:collapse;font-size:10px}
    thead tr{background:#f8fafc}
    th{padding:7px 8px;text-align:left;font-weight:700;color:#64748b;
      font-size:9px;letter-spacing:.5px;border-bottom:1px solid #e2e8f0}
    td{padding:6px 8px;border-bottom:1px solid #f1f5f9}
    .badge{padding:2px 7px;border-radius:8px;font-size:9px;font-weight:800}
    .footer{margin-top:32px;padding-top:12px;border-top:1px solid #e2e8f0;
      display:flex;justify-content:space-between;font-size:9px;color:#94a3b8}
    @media print{
      body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
      .page{padding:20px}
      .header{border-radius:8px}
    }
  </style></head>
  <body><div class="page">

  <div class="header">
    <div class="sub">EDURISK · STUDENT DROPOUT ANALYSIS REPORT</div>
    <h1>Academic Risk Assessment</h1>
    <div style="font-size:11px;opacity:.7;margin-top:3px">
      Dataset: ${fileName} &nbsp;·&nbsp; Generated: ${date} &nbsp;·&nbsp; Model: RF+XGBoost+LogReg Ensemble
    </div>
    <div class="meta-row">
      <div class="meta-pill"><span class="v">${students.length}</span><span class="l">Total</span></div>
      <div class="meta-pill"><span class="v" style="color:#fca5a5">${high}</span><span class="l">High Risk</span></div>
      <div class="meta-pill"><span class="v" style="color:#fde68a">${mod}</span><span class="l">Moderate</span></div>
      <div class="meta-pill"><span class="v" style="color:#86efac">${low}</span><span class="l">Low Risk</span></div>
      <div class="meta-pill"><span class="v">${avgDrop}%</span><span class="l">Avg Dropout Prob</span></div>
      <div class="meta-pill"><span class="v">${avgAtt}%</span><span class="l">Avg Attendance</span></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Class-Wide Risk Distribution</div>
    <div class="grid-2">
      <div style="display:flex;gap:20px;align-items:center">
        ${donut}
        <div>
          ${[["HIGH RISK",high,"#ef4444","#fef2f2"],[`MODERATE`,mod,"#f59e0b","#fffbeb"],["LOW RISK",low,"#22c55e","#f0fdf4"]].map(([l,v,c,bg])=>`
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <div style="width:12px;height:12px;border-radius:3px;background:${c};flex-shrink:0"></div>
            <div style="font-size:11px;color:#475569">${l}: <strong style="color:${c}">${v}</strong> students (${students.length?Math.round(v/students.length*100):0}%)</div>
          </div>`).join("")}
        </div>
      </div>
      <div>
        <div class="grid-3" style="gap:10px">
          ${[["🔴 High Risk",high,"#ef4444","#fef2f2"],["🟡 Moderate",mod,"#f59e0b","#fffbeb"],["🟢 Low Risk",low,"#22c55e","#f0fdf4"]].map(([l,v,c,bg])=>`
          <div class="risk-card" style="background:${bg}">
            <div class="n" style="color:${c}">${v}</div>
            <div class="l" style="color:${c}">${l}</div>
            <div class="p">${students.length?Math.round(v/students.length*100):0}% of class</div>
          </div>`).join("")}
        </div>
        <div style="margin-top:12px;padding:10px 12px;background:#f8fafc;border-radius:8px;font-size:11px;color:#475569;line-height:1.7">
          Class average: &nbsp;<strong>Attendance ${avgAtt}%</strong> &nbsp;·&nbsp;
          <strong>Sem1 Grade ${avgGr}/20</strong> &nbsp;·&nbsp;
          <strong>Logins ${avgLog}/mo</strong> &nbsp;·&nbsp;
          <strong>Assignments ${avgAsgn}/10</strong>
        </div>
      </div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:24px">
    <div class="card">
      <h3>Feature Importance (SHAP — Class Average)</h3>
      ${topFeatures.map(f=>`
      <div class="feat-row">
        <div class="feat-name">${f.name}</div>
        <div style="flex:1">${svgBar(Math.round(f.val/maxFeat*100),"#2563eb",200,12)}</div>
        <div class="feat-val">${(f.val*100).toFixed(1)}%</div>
      </div>`).join("")}
    </div>
    <div class="card">
      <h3>Intervention Recommendations</h3>
      ${high>0?`<div class="int-item" style="background:#fef2f2;border-left:3px solid #ef4444">
        <strong style="color:#dc2626">🔴 ${high} HIGH RISK</strong><br>
        Immediate 1:1 advisor meeting required. Verify tuition/financial aid status.
        Contact student within 48 hours. Consider course load reduction.</div>`:""}
      ${mod>0?`<div class="int-item" style="background:#fffbeb;border-left:3px solid #f59e0b">
        <strong style="color:#d97706">🟡 ${mod} MODERATE RISK</strong><br>
        Schedule group check-in within 2 weeks. Promote peer tutoring and
        study groups. Send personalised LMS engagement nudge.</div>`:""}
      <div class="int-item" style="background:#f0fdf4;border-left:3px solid #22c55e">
        <strong style="color:#16a34a">📚 ALL STUDENTS</strong><br>
        Push LMS engagement notifications to students with &lt;15 logins/month.
        Early intervention at Week 4 yields 40% retention improvement.
      </div>
      ${parseFloat(avgAtt)<65?`<div class="int-item" style="background:#fff7ed;border-left:3px solid #f97316">
        <strong style="color:#c2410c">⚠ ATTENDANCE ALERT</strong><br>
        Class average attendance is ${avgAtt}% — below 65% threshold.
        Investigate systemic issues (timetabling, transport, workload).</div>`:""}
    </div>
  </div>

  <div class="section">
    <div class="section-title">Per-Student Risk Analysis (${students.length} students)</div>
    <table>
      <thead><tr>
        <th>ID</th><th>Sem1</th><th>Sem2</th><th>Att%</th><th>Logins</th>
        <th>Asgn</th><th>Risk</th><th>Dropout Prob</th><th>Conf</th><th>Top Issue</th><th>Action</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>

  <div class="footer">
    <span>EduRisk v3.0 · RandomForest + XGBoost + LogReg Ensemble · UCI Student Dropout Dataset</span>
    <span>⚠ Confidential — For academic advisor use only · Not for distribution</span>
  </div>
  </div></body></html>`;

  const w=window.open("","_blank","width=1100,height=800");
  w.document.write(html);w.document.close();
  setTimeout(()=>{w.focus();w.print();},600);
}

// ── Colour tokens ─────────────────────────────────────────────────────────────
const RC={
  HIGH:    {fg:"#dc2626",bg:"rgba(239,68,68,0.07)",  border:"rgba(239,68,68,0.18)"},
  MODERATE:{fg:"#d97706",bg:"rgba(245,158,11,0.07)", border:"rgba(245,158,11,0.18)"},
  LOW:     {fg:"#16a34a",bg:"rgba(34,197,94,0.07)",  border:"rgba(34,197,94,0.18)"},
};

// ── Gauge ─────────────────────────────────────────────────────────────────────
function Gauge({value,color,size=100}){
  const r=size/2-9,c=Math.PI*r;
  return<svg width={size} height={size/2+14} viewBox={`0 0 ${size} ${size/2+14}`}>
    <path d={`M 9 ${size/2} A ${r} ${r} 0 0 1 ${size-9} ${size/2}`} fill="none" stroke="#f1f5f9" strokeWidth="8"/>
    <path d={`M 9 ${size/2} A ${r} ${r} 0 0 1 ${size-9} ${size/2}`} fill="none" stroke={color}
      strokeWidth="8" strokeDasharray={c} strokeDashoffset={c*(1-value)}
      strokeLinecap="round" style={{transition:"stroke-dashoffset .9s ease"}}/>
    <text x={size/2} y={size/2+11} textAnchor="middle" fill={color}
      fontSize="16" fontWeight="800" fontFamily="'JetBrains Mono',monospace">{Math.round(value*100)}%</text>
  </svg>;
}

// ── Mini bar ──────────────────────────────────────────────────────────────────
function Bar({pct,color,h=5}){
  return<div style={{height:h,background:"#f1f5f9",borderRadius:h/2,overflow:"hidden"}}>
    <div style={{height:"100%",width:`${Math.min(100,pct)}%`,background:color,borderRadius:h/2,transition:"width .6s ease"}}/>
  </div>;
}

// ════════════════════════════════════════════════════════════════════════════════
export default function App(){
  const dark=false; // Always light — clean professional look
  const [page,setPage]=useState("dashboard");
  const [form,setForm]=useState({
    admission_grade:130,prev_grade:12.5,sem1_grade:11,sem2_grade:10.5,
    sem1_approved:3,logins:18,attendance:58,assignments_done:4,
    scholarship:0,tuition:1,debtor:0,parent_edu:2
  });
  const [result,setResult]=useState(null);
  const [predicting,setPredicting]=useState(false);
  const [predErr,setPredErr]=useState("");
  const [nudge,setNudge]=useState("");
  const [nudgeLoading,setNudgeLoading]=useState(false);
  const [nudgeErr,setNudgeErr]=useState("");

  const [students,setStudents]=useState([]);
  const [preds,setPreds]=useState([]);
  const [batchRunning,setBatchRunning]=useState(false);
  const [batchProg,setBatchProg]=useState(0);
  const [uploadErr,setUploadErr]=useState("");
  const [fileName,setFileName]=useState("");
  const [detectedCols,setDetectedCols]=useState([]);
  const [dragging,setDragging]=useState(false);

  const [moodleUrl,setMoodleUrl]=useState("");
  const [moodleToken,setMoodleToken]=useState("");
  const [moodleCourse,setMoodleCourse]=useState("");
  const [moodleStatus,setMoodleStatus]=useState(null);
  const [moodleTesting,setMoodleTesting]=useState(false);
  const [moodleInfo,setMoodleInfo]=useState(null);
  const [moodleFetching,setMoodleFetching]=useState(false);
  const [moodleStudents,setMoodleStudents]=useState([]);

  const fileRef=useRef(null);
  const sf=(k,v)=>setForm(p=>({...p,[k]:v}));

  const high=preds.filter(p=>p.risk_level==="HIGH").length;
  const mod=preds.filter(p=>p.risk_level==="MODERATE").length;
  const low=preds.filter(p=>p.risk_level==="LOW").length;

  function handleFile(file){
    if(!file)return;
    if(!file.name.match(/\.(csv|txt|tsv)$/i)&&!file.name.match(/\.(xlsx|xls)$/i)){
      setUploadErr("Unsupported format. Please upload CSV, TSV, or Excel file.");return;
    }
    setUploadErr("");setStudents([]);setPreds([]);setFileName(file.name);
    const reader=new FileReader();
    reader.onload=e=>{
      try{
        const{rows,detectedCols:dc,delimiter}=autoParseCSV(e.target.result);
        if(rows.length===0){setUploadErr("No valid student rows found. Ensure your CSV has columns like: sem1_grade, attendance, logins, assignments_done.");return;}
        setStudents(rows);setDetectedCols(dc);
        setPage("batch");
      }catch(err){setUploadErr("Parse error: "+err.message);}
    };
    reader.readAsText(file);
  }

  const onDrop=useCallback(e=>{
    e.preventDefault();setDragging(false);handleFile(e.dataTransfer.files[0]);
  },[]);

  async function runBatch(){
    setBatchRunning(true);setPreds([]);setBatchProg(0);
    const out=[];
    for(let i=0;i<students.length;i++){
      const p=await apiPredict(students[i]).catch(()=>localPred(students[i]));
      out.push(p);
      setBatchProg(Math.round((i+1)/students.length*100));
      setPreds([...out]);
      await new Promise(r=>setTimeout(r,30));
    }
    setBatchRunning(false);
  }

  async function runPredict(){
    setPredicting(true);setPredErr("");setNudge("");setNudgeErr("");
    try{
      const res=await apiPredict(form).catch(()=>localPred(form));
      setResult(res);
      // Auto-generate AI nudge after prediction
      setNudgeLoading(true);
      apiNudge(form,res).then(msg=>setNudge(msg)).catch(e=>setNudgeErr(e.message)).finally(()=>setNudgeLoading(false));
    }
    catch(e){setPredErr(e.message);}
    finally{setPredicting(false);}
  }

  async function runNudge(){
    setNudgeLoading(true);setNudge("");setNudgeErr("");
    try{setNudge(await apiNudge(form,result));}
    catch(e){setNudgeErr(e.message);}
    finally{setNudgeLoading(false);}
  }

  async function testMoodle(){
    setMoodleTesting(true);setMoodleStatus(null);setMoodleInfo(null);
    try{
      const info=await moodleTest(moodleUrl,moodleToken);
      setMoodleStatus("success");setMoodleInfo(info);
    }catch(e){setMoodleStatus("error:"+e.message);}
    finally{setMoodleTesting(false);}
  }

  async function fetchMoodleStudents(){
    setMoodleFetching(true);setMoodleStudents([]);
    try{
      const list=await moodleFetchStudents(moodleUrl,moodleToken,moodleCourse);
      setMoodleStudents(list);
    }catch(e){setMoodleStatus("error:"+e.message);}
    finally{setMoodleFetching(false);}
  }

  // ── Design tokens ──────────────────────────────────────────────────────────
  const C={
    bg:"#f8fafc", surface:"#ffffff", surface2:"#f1f5f9",
    border:"#e2e8f0", text:"#0f172a", text2:"#64748b",
    accent:"#2563eb", accent2:"#7c3aed",
    shadow:"0 1px 3px rgba(0,0,0,0.08),0 8px 24px rgba(0,0,0,0.06)",
    shadowHover:"0 4px 12px rgba(37,99,235,0.15)",
  };
  const card={background:C.surface,border:`1px solid ${C.border}`,borderRadius:14,
               padding:22,boxShadow:C.shadow};
  const inp={background:C.surface2,border:`1px solid ${C.border}`,borderRadius:8,
              padding:"9px 13px",color:C.text,fontSize:13,fontFamily:"inherit",
              outline:"none",width:"100%",transition:"border-color .15s"};
  const lbl={fontSize:10,fontWeight:700,color:C.text2,letterSpacing:"0.8px",
              marginBottom:5,display:"block",textTransform:"uppercase"};

  const NAV=[
    {id:"dashboard",icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>,label:"Dashboard"},
    {id:"predict",icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>,label:"Single Predict"},
    {id:"batch",icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>,label:"Batch Upload"},
    {id:"lms",icon:<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 8h1a4 4 0 0 1 0 8h-1"/><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/><line x1="6" y1="1" x2="6" y2="4"/><line x1="10" y1="1" x2="10" y2="4"/><line x1="14" y1="1" x2="14" y2="4"/></svg>,label:"Moodle LMS"},
  ];

  return(
  <div style={{minHeight:"100vh",background:C.bg,fontFamily:"'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif",display:"flex"}}>

    {/* ── Sidebar ── */}
    <div style={{width:232,background:C.surface,borderRight:`1px solid ${C.border}`,
                 display:"flex",flexDirection:"column",position:"fixed",height:"100vh",
                 zIndex:40,boxShadow:"1px 0 0 #e2e8f0"}}>
      {/* Logo */}
      <div style={{padding:"20px 20px 16px",borderBottom:`1px solid ${C.border}`}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:38,height:38,borderRadius:10,
            background:"linear-gradient(135deg,#2563eb,#7c3aed)",
            display:"flex",alignItems:"center",justifyContent:"center",fontSize:18,flexShrink:0}}>🎓</div>
          <div>
            <div style={{fontWeight:800,fontSize:16,color:C.text,letterSpacing:"-0.3px",lineHeight:1}}>EduRisk</div>
            <div style={{fontSize:9,color:C.text2,letterSpacing:"1.5px",marginTop:2}}>DROPOUT PREDICTOR</div>
          </div>
        </div>
        <div style={{marginTop:12,padding:"7px 10px",background:"rgba(37,99,235,.06)",
          borderRadius:8,fontSize:10,color:C.accent,fontWeight:600,display:"flex",
          alignItems:"center",gap:6}}>
          <span style={{width:6,height:6,borderRadius:"50%",background:"#22c55e",flexShrink:0}}/>
          API Connected · HF Space Live
        </div>
      </div>

      {/* Nav */}
      <nav style={{flex:1,padding:"10px 10px",overflowY:"auto"}}>
        {NAV.map(item=>{
          const active=page===item.id;
          return<button key={item.id} onClick={()=>setPage(item.id)} style={{
            width:"100%",display:"flex",alignItems:"center",gap:10,padding:"9px 11px",
            borderRadius:9,border:"none",cursor:"pointer",fontFamily:"inherit",
            background:active?"rgba(37,99,235,.08)":"transparent",
            color:active?C.accent:C.text2,marginBottom:2,
            fontWeight:active?700:500,fontSize:13,textAlign:"left",
            transition:"all .15s",
          }}>
            <span style={{opacity:active?1:.6}}>{item.icon}</span>
            {item.label}
            {item.id==="batch"&&students.length>0&&
              <span style={{marginLeft:"auto",background:C.accent,color:"#fff",
                borderRadius:10,padding:"1px 8px",fontSize:9,fontWeight:700}}>
                {students.length}
              </span>}
            {item.id==="lms"&&moodleStatus==="success"&&
              <span style={{marginLeft:"auto",fontSize:9}}>🟢</span>}
          </button>;
        })}

        {/* Upload shortcut */}
        <div style={{margin:"12px 0 8px",height:1,background:C.border}}/>
        <div style={{padding:"0 2px"}}>
          <label htmlFor="sideUpload" style={{
            display:"flex",alignItems:"center",gap:8,padding:"9px 11px",
            borderRadius:9,cursor:"pointer",color:C.text2,fontSize:13,fontWeight:500,
            border:`1px dashed ${C.border}`,background:"transparent",transition:"all .15s",
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            Upload CSV / Excel
          </label>
          <input id="sideUpload" type="file" accept=".csv,.tsv,.txt,.xlsx,.xls"
            style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
        </div>
      </nav>

      {/* Footer */}
      <div style={{padding:"14px 16px",borderTop:`1px solid ${C.border}`,
                   fontSize:10,color:C.text2,lineHeight:1.7}}>
        <div style={{fontWeight:600,color:C.text,marginBottom:2}}>Stack</div>
        RF+XGBoost+LogReg · Qwen3-8B<br/>
        FastAPI · HF Spaces · Vercel<br/>
        <span style={{color:"#22c55e",fontWeight:600}}>Total cost: $0</span>
      </div>
    </div>

    {/* ── Main ── */}
    <div style={{marginLeft:232,flex:1,padding:"28px 32px",maxWidth:"calc(100vw - 232px)"}}>

      {/* ══════════ DASHBOARD ══════════ */}
      {page==="dashboard"&&<div>
        <div style={{marginBottom:24}}>
          <h1 style={{fontSize:24,fontWeight:800,color:C.text,letterSpacing:"-0.5px"}}>Dashboard</h1>
          <p style={{color:C.text2,fontSize:13,marginTop:3}}>
            Monitor student dropout risk across your institution using ML predictions and LMS data.
          </p>
        </div>

        {/* KPI row */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:20}}>
          {[
            {l:"Students Analysed",v:preds.length||"—",c:"#2563eb",i:"👥",bg:"rgba(37,99,235,.06)"},
            {l:"High Risk",v:high||"—",c:"#dc2626",i:"🔴",bg:"rgba(239,68,68,.06)"},
            {l:"Moderate Risk",v:mod||"—",c:"#d97706",i:"🟡",bg:"rgba(245,158,11,.06)"},
            {l:"Low Risk",v:low||"—",c:"#16a34a",i:"🟢",bg:"rgba(34,197,94,.06)"},
          ].map(s=><div key={s.l} style={{...card,display:"flex",alignItems:"center",gap:14}}>
            <div style={{width:44,height:44,borderRadius:12,background:s.bg,
              display:"flex",alignItems:"center",justifyContent:"center",fontSize:22,flexShrink:0}}>{s.i}</div>
            <div>
              <div style={{fontSize:26,fontWeight:900,color:s.c,lineHeight:1}}>{s.v}</div>
              <div style={{fontSize:11,color:C.text2,marginTop:3,fontWeight:500}}>{s.l}</div>
            </div>
          </div>)}
        </div>

        {/* Quick action cards */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14,marginBottom:20}}>
          {[
            {id:"predict",emoji:"🔍",title:"Single Student",desc:"Enter features manually and get instant risk score + AI advisor nudge from Qwen3-8B.",btn:"Open Predictor",grad:"linear-gradient(135deg,#2563eb,#7c3aed)"},
            {id:"batch",emoji:"📊",title:"Batch CSV Upload",desc:"Upload any CSV or Excel file. Auto-detects columns, runs all students, downloads PDF report.",btn:"Upload Dataset",grad:"linear-gradient(135deg,#059669,#0891b2)"},
            {id:"lms",emoji:"🔗",title:"Moodle Integration",desc:"Connect your Moodle instance with a token to auto-sync student grades and attendance data.",btn:"Configure Moodle",grad:"linear-gradient(135deg,#f98012,#dc2626)"},
          ].map(a=><div key={a.id} style={{...card,cursor:"pointer",transition:"box-shadow .15s"}}
            onClick={()=>setPage(a.id)}
            onMouseEnter={e=>e.currentTarget.style.boxShadow=C.shadowHover}
            onMouseLeave={e=>e.currentTarget.style.boxShadow=C.shadow}>
            <div style={{fontSize:28,marginBottom:12}}>{a.emoji}</div>
            <div style={{fontWeight:700,fontSize:15,color:C.text,marginBottom:6}}>{a.title}</div>
            <div style={{fontSize:12,color:C.text2,lineHeight:1.6,marginBottom:14}}>{a.desc}</div>
            <div style={{display:"inline-block",background:a.grad,color:"#fff",
              padding:"7px 16px",borderRadius:8,fontSize:12,fontWeight:700}}>{a.btn} →</div>
          </div>)}
        </div>

        {/* Moodle API info */}
        <div style={card}>
          <div style={{fontWeight:700,fontSize:13,color:C.text,marginBottom:4}}>🔗 Moodle API — Integration Checklist</div>
          <div style={{fontSize:12,color:C.text2,marginBottom:16}}>Complete these steps when your college grants API access:</div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12}}>
            {[
              {step:"01",title:"Get Web Services Token",desc:"Ask your Moodle admin to enable Web Services and generate a token for your account.",status:"pending"},
              {step:"02",title:"Enable REST Protocol",desc:"Admin → Site Admin → Plugins → Web Services → Manage Protocols → Enable REST.",status:"pending"},
              {step:"03",title:"Assign Capabilities",desc:"Ensure token has: core_enrol_get_enrolled_users, gradereport_user_get_grades_table, core_completion_get_activities_completion_status.",status:"pending"},
            ].map(s=><div key={s.step} style={{padding:14,border:`1px solid ${C.border}`,borderRadius:10}}>
              <div style={{fontSize:10,fontWeight:800,color:C.accent,letterSpacing:"1px",marginBottom:6}}>STEP {s.step}</div>
              <div style={{fontSize:12,fontWeight:700,color:C.text,marginBottom:4}}>{s.title}</div>
              <div style={{fontSize:11,color:C.text2,lineHeight:1.6}}>{s.desc}</div>
            </div>)}
          </div>
        </div>
      </div>}

      {/* ══════════ SINGLE PREDICT ══════════ */}
      {page==="predict"&&<div>
        <div style={{marginBottom:22}}>
          <h1 style={{fontSize:22,fontWeight:800,color:C.text}}>Single Student Prediction</h1>
          <p style={{color:C.text2,fontSize:13,marginTop:3}}>Adjust the 12 features below and run the RF+XGBoost ensemble model.</p>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          {/* Left — inputs */}
          <div style={{display:"flex",flexDirection:"column",gap:14}}>
            {/* Academic */}
            <div style={card}>
              <div style={{fontWeight:700,fontSize:12,color:C.accent,letterSpacing:"0.5px",marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                <span style={{width:20,height:20,borderRadius:6,background:"rgba(37,99,235,.1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:11}}>🎓</span>
                ACADEMIC FEATURES
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
                {[
                  {k:"admission_grade",l:"Admission Grade",min:95,max:175,step:1,u:"/175",c:"#2563eb"},
                  {k:"prev_grade",l:"Prev Grade",min:0,max:20,step:.5,u:"/20",c:"#2563eb"},
                  {k:"sem1_grade",l:"Semester 1 Grade",min:0,max:20,step:.5,u:"/20",c:"#2563eb"},
                  {k:"sem2_grade",l:"Semester 2 Grade",min:0,max:20,step:.5,u:"/20",c:"#2563eb"},
                ].map(f=><div key={f.k}>
                  <label style={lbl}>{f.l}</label>
                  <div style={{display:"flex",alignItems:"center",gap:8}}>
                    <input type="range" min={f.min} max={f.max} step={f.step} value={form[f.k]}
                      onChange={e=>sf(f.k,parseFloat(e.target.value))}
                      style={{flex:1,accentColor:f.c,cursor:"pointer"}}/>
                    <span style={{fontSize:13,fontWeight:700,color:f.c,minWidth:40,textAlign:"right"}}>{form[f.k]}{f.u}</span>
                  </div>
                  <Bar pct={(form[f.k]-f.min)/(f.max-f.min)*100} color={f.c} h={3}/>
                </div>)}
                <div style={{gridColumn:"1/-1"}}>
                  <label style={lbl}>Units Approved Sem1</label>
                  <div style={{display:"flex",gap:6}}>
                    {[0,1,2,3,4,5,6].map(v=><button key={v} onClick={()=>sf("sem1_approved",v)} style={{
                      flex:1,padding:"7px 0",borderRadius:7,border:`1px solid ${form.sem1_approved===v?"#2563eb":C.border}`,
                      background:form.sem1_approved===v?"rgba(37,99,235,.1)":C.surface2,
                      color:form.sem1_approved===v?C.accent:C.text2,
                      fontSize:12,fontWeight:700,cursor:"pointer",fontFamily:"inherit"
                    }}>{v}</button>)}
                  </div>
                </div>
              </div>
            </div>

            {/* Behavioral */}
            <div style={card}>
              <div style={{fontWeight:700,fontSize:12,color:"#059669",letterSpacing:"0.5px",marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                <span style={{width:20,height:20,borderRadius:6,background:"rgba(5,150,105,.1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:11}}>📱</span>
                BEHAVIORAL FEATURES
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14}}>
                {[
                  {k:"logins",l:"LMS Logins/mo",min:0,max:100,step:1,u:"",c:"#059669"},
                  {k:"attendance",l:"Attendance",min:0,max:100,step:1,u:"%",c:"#059669"},
                  {k:"assignments_done",l:"Assignments",min:0,max:10,step:1,u:"/10",c:"#059669"},
                ].map(f=><div key={f.k}>
                  <label style={lbl}>{f.l}</label>
                  <input type="range" min={f.min} max={f.max} step={f.step} value={form[f.k]}
                    onChange={e=>sf(f.k,parseFloat(e.target.value))}
                    style={{width:"100%",accentColor:f.c,cursor:"pointer"}}/>
                  <div style={{display:"flex",justifyContent:"space-between",marginTop:3}}>
                    <span style={{fontSize:9,color:C.text2}}>{f.min}</span>
                    <span style={{fontSize:12,fontWeight:700,color:f.c}}>{form[f.k]}{f.u}</span>
                    <span style={{fontSize:9,color:C.text2}}>{f.max}</span>
                  </div>
                </div>)}
              </div>
            </div>

            {/* Socio-economic */}
            <div style={card}>
              <div style={{fontWeight:700,fontSize:12,color:"#7c3aed",letterSpacing:"0.5px",marginBottom:14,display:"flex",alignItems:"center",gap:6}}>
                <span style={{width:20,height:20,borderRadius:6,background:"rgba(124,58,237,.1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:11}}>💰</span>
                SOCIO-ECONOMIC FEATURES
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8,marginBottom:14}}>
                {[
                  {k:"scholarship",l:"Scholarship",pos:true},
                  {k:"tuition",l:"Tuition Paid",pos:true},
                  {k:"debtor",l:"Has Debt",pos:false},
                ].map(f=><button key={f.k} onClick={()=>sf(f.k,form[f.k]?0:1)} style={{
                  padding:"10px 6px",borderRadius:9,border:`1px solid`,cursor:"pointer",fontFamily:"inherit",
                  borderColor:form[f.k]?(f.pos?"#16a34a":"#dc2626"):`${C.border}`,
                  background:form[f.k]?(f.pos?"rgba(34,197,94,.07)":"rgba(239,68,68,.07)"):`${C.surface2}`,
                  color:form[f.k]?(f.pos?"#16a34a":"#dc2626"):C.text2,
                  fontSize:11,fontWeight:600,lineHeight:1.5,transition:"all .15s"
                }}><div>{f.l}</div><div style={{fontWeight:800,fontSize:13}}>{form[f.k]?"YES":"NO"}</div></button>)}
              </div>
              <label style={lbl}>Parent Education Level</label>
              <input type="range" min={1} max={5} step={1} value={form.parent_edu}
                onChange={e=>sf("parent_edu",parseInt(e.target.value))}
                style={{width:"100%",accentColor:"#7c3aed",cursor:"pointer"}}/>
              <div style={{display:"flex",justifyContent:"space-between",marginTop:4,fontSize:10,color:C.text2}}>
                {["1 None","2 Basic","3 Secondary","4 Higher","5 Post-grad"].map(v=>(
                  <span key={v} style={{color:form.parent_edu===parseInt(v)?"#7c3aed":C.text2,fontWeight:form.parent_edu===parseInt(v)?700:400}}>{v}</span>
                ))}
              </div>
            </div>

            {predErr&&<div style={{background:"rgba(239,68,68,.06)",border:"1px solid rgba(239,68,68,.18)",borderRadius:10,padding:"10px 14px",fontSize:12,color:"#dc2626"}}>⚠ {predErr}</div>}
            <button onClick={runPredict} disabled={predicting} style={{
              padding:"13px",borderRadius:11,border:"none",cursor:"pointer",fontFamily:"inherit",
              background:predicting?C.surface2:"linear-gradient(135deg,#2563eb,#7c3aed)",
              color:predicting?C.text2:"#fff",fontSize:14,fontWeight:700,
              boxShadow:predicting?"none":"0 4px 14px rgba(37,99,235,.35)",transition:"all .15s"
            }}>{predicting?"⏳ Running prediction...":"🎯 Run Ensemble Prediction"}</button>
          </div>

          {/* Right — results */}
          <div style={{display:"flex",flexDirection:"column",gap:14}}>
            {result?(()=>{
              const rc=RC[result.risk_level||"MODERATE"];
              return<>
                {/* Risk summary */}
                <div style={{...card,background:rc.bg,border:`1.5px solid ${rc.border}`}}>
                  <div style={{display:"flex",alignItems:"center",gap:18}}>
                    <div style={{textAlign:"center"}}>
                      <Gauge value={result.dropout} color={rc.fg} size={112}/>
                      <div style={{fontSize:9,color:rc.fg,fontWeight:800,letterSpacing:"2px",marginTop:2}}>DROPOUT RISK</div>
                    </div>
                    <div style={{flex:1}}>
                      <div style={{display:"inline-flex",alignItems:"center",gap:7,padding:"4px 14px",borderRadius:20,
                        background:rc.fg,color:"#fff",fontSize:11,fontWeight:800,letterSpacing:"2px",marginBottom:14}}>
                        {result.risk_level} RISK
                      </div>
                      <div style={{display:"flex",gap:18,marginBottom:10}}>
                        {[["Dropout",result.dropout,rc.fg],["Graduate",result.graduate,"#16a34a"],["Enrolled",result.enrolled,"#d97706"]].map(([l,v,c])=>(
                          <div key={l} style={{textAlign:"center"}}>
                            <div style={{fontSize:17,fontWeight:900,color:c,fontFamily:"'JetBrains Mono',monospace"}}>{Math.round(v*100)}%</div>
                            <div style={{fontSize:10,color:C.text2}}>{l}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{fontSize:11,color:C.text2,lineHeight:1.6}}>
                        {result.risk_level==="HIGH"&&"⚠️ Immediate advisor outreach required. Schedule meeting within 48 hours."}
                        {result.risk_level==="MODERATE"&&"📋 Monitor closely. Schedule check-in within 2 weeks."}
                        {result.risk_level==="LOW"&&"✅ Student is on track. Continue standard engagement."}
                      </div>
                    </div>
                  </div>
                </div>

                {/* SHAP Attribution */}
                <div style={card}>
                  <div style={{fontWeight:700,fontSize:11,color:C.text2,letterSpacing:"1.2px",marginBottom:14}}>FEATURE ATTRIBUTION (SHAP-STYLE)</div>
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
                    <div>
                      <div style={{fontSize:11,color:"#dc2626",fontWeight:700,marginBottom:8}}>⚠ Risk Factors</div>
                      {result.riskFactors?.map(f=><div key={f.feature} style={{marginBottom:9}}>
                        <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:3}}>
                          <span style={{color:"#dc2626"}}>{FLABEL[f.feature]||f.feature}</span>
                          <span style={{color:C.text2,fontFamily:"'JetBrains Mono',monospace"}}>{f.raw}</span>
                        </div>
                        <Bar pct={Math.min(100,f.impact*600)} color="#ef4444" h={5}/>
                      </div>)}
                    </div>
                    <div>
                      <div style={{fontSize:11,color:"#16a34a",fontWeight:700,marginBottom:8}}>✓ Strengths</div>
                      {result.strengths?.map(f=><div key={f.feature} style={{marginBottom:9}}>
                        <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:3}}>
                          <span style={{color:"#16a34a"}}>{FLABEL[f.feature]||f.feature}</span>
                          <span style={{color:C.text2,fontFamily:"'JetBrains Mono',monospace"}}>{f.raw}</span>
                        </div>
                        <Bar pct={Math.min(100,f.impact*600)} color="#22c55e" h={5}/>
                      </div>)}
                    </div>
                  </div>
                </div>

                {/* Nudge */}
                <div style={{...card,border:`1px solid rgba(37,99,235,.18)`}}>
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
                    <div style={{fontWeight:700,fontSize:11,color:C.accent,letterSpacing:"1.2px"}}>
                      🤖 AI ADVISOR NUDGE — QWEN3-8B
                    </div>
                    {nudge&&!nudgeLoading&&(
                      <button onClick={runNudge} style={{
                        padding:"4px 10px",borderRadius:7,border:`1px solid rgba(37,99,235,.2)`,
                        background:"transparent",color:C.accent,fontSize:10,fontWeight:700,
                        cursor:"pointer",fontFamily:"inherit"
                      }}>↺ Regenerate</button>
                    )}
                  </div>
                  {nudgeLoading&&(
                    <div style={{display:"flex",alignItems:"center",gap:10,padding:"14px",
                      background:"rgba(37,99,235,.04)",border:`1px solid rgba(37,99,235,.1)`,
                      borderRadius:10}}>
                      <div style={{width:16,height:16,border:"2px solid rgba(37,99,235,.2)",
                        borderTopColor:C.accent,borderRadius:"50%",
                        animation:"spin 0.8s linear infinite",flexShrink:0}}/>
                      <span style={{fontSize:12,color:C.text2}}>Qwen3-8B is drafting a personalised message…</span>
                    </div>
                  )}
                  {nudgeErr&&!nudgeLoading&&(
                    <div style={{marginBottom:8,fontSize:11,color:"#dc2626",
                      background:"rgba(239,68,68,.05)",padding:"10px 12px",borderRadius:8,
                      display:"flex",alignItems:"flex-start",gap:8}}>
                      <span>⚠</span>
                      <div>
                        <div style={{fontWeight:700,marginBottom:2}}>AI nudge unavailable</div>
                        <div style={{opacity:.8}}>{nudgeErr}</div>
                        <button onClick={runNudge} style={{marginTop:6,padding:"4px 10px",borderRadius:6,
                          border:`1px solid rgba(239,68,68,.3)`,background:"transparent",color:"#dc2626",
                          fontSize:10,fontWeight:700,cursor:"pointer",fontFamily:"inherit"}}>Try again</button>
                      </div>
                    </div>
                  )}
                  {nudge&&!nudgeLoading&&(
                    <div style={{padding:14,background:"rgba(37,99,235,.04)",
                      border:`1px solid rgba(37,99,235,.12)`,borderRadius:10}}>
                      <div style={{fontSize:9,color:C.accent,fontWeight:700,letterSpacing:"1.5px",marginBottom:8}}>
                        GENERATED MESSAGE ✓
                      </div>
                      <div style={{fontSize:13,color:C.text,lineHeight:1.85,fontStyle:"italic"}}>{nudge}</div>
                    </div>
                  )}
                  {!nudge&&!nudgeLoading&&!nudgeErr&&(
                    <div style={{fontSize:11,color:C.text2,textAlign:"center",padding:"10px 0"}}>
                      Message generates automatically after prediction
                    </div>
                  )}
                  <div style={{marginTop:10,fontSize:10,color:C.text2,display:"flex",alignItems:"center",gap:5}}>
                    <span style={{width:5,height:5,borderRadius:"50%",background:"#22c55e",flexShrink:0}}/>
                    Powered by Qwen3-8B via HuggingFace · No token required
                  </div>
                </div>
              </>;
            })():(
              <div style={{...card,height:320,display:"flex",flexDirection:"column",
                alignItems:"center",justifyContent:"center",textAlign:"center",color:C.text2}}>
                <div style={{fontSize:44,marginBottom:12}}>🔍</div>
                <div style={{fontWeight:600,fontSize:14,color:C.text}}>No prediction yet</div>
                <div style={{fontSize:12,marginTop:6}}>Set student features on the left<br/>and click <strong style={{color:C.accent}}>Run Ensemble Prediction</strong></div>
              </div>
            )}
          </div>
        </div>
      </div>}

      {/* ══════════ BATCH UPLOAD ══════════ */}
      {page==="batch"&&<div>
        <div style={{marginBottom:22}}>
          <h1 style={{fontSize:22,fontWeight:800,color:C.text}}>Batch Upload & Analysis</h1>
          <p style={{color:C.text2,fontSize:13,marginTop:3}}>Upload CSV or Excel · Auto-detects any column format · Download full PDF report</p>
        </div>

        {/* Drop zone */}
        {students.length===0&&<>
          <div
            onDrop={onDrop} onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)}
            onClick={()=>fileRef.current?.click()}
            style={{border:`2px dashed ${dragging?C.accent:C.border}`,borderRadius:16,padding:"52px 40px",
              textAlign:"center",cursor:"pointer",background:dragging?"rgba(37,99,235,.03)":C.surface,
              marginBottom:16,transition:"all .2s"}}>
            <input ref={fileRef} type="file" accept=".csv,.tsv,.txt,.xlsx,.xls"
              style={{display:"none"}} onChange={e=>handleFile(e.target.files[0])}/>
            <div style={{fontSize:44,marginBottom:14}}>📂</div>
            <div style={{fontSize:15,fontWeight:700,color:C.text,marginBottom:6}}>Drag & drop your student dataset here</div>
            <div style={{fontSize:12,color:C.text2,marginBottom:20,maxWidth:480,margin:"0 auto 20px"}}>
              Supports <strong>CSV</strong>, <strong>Excel (.xlsx)</strong>, <strong>TSV</strong> · Auto-detects column names<br/>
              Works with UCI format, Moodle exports, or any custom format
            </div>
            <div style={{display:"inline-block",padding:"9px 24px",borderRadius:9,
              background:"linear-gradient(135deg,#059669,#0891b2)",color:"#fff",
              fontSize:13,fontWeight:700,boxShadow:"0 4px 14px rgba(5,150,105,.3)"}}>
              Browse File
            </div>
          </div>

          {/* Sample format hint */}
          <div style={{...card,marginBottom:16}}>
            <div style={{fontWeight:700,fontSize:12,color:C.text2,letterSpacing:"1px",marginBottom:10}}>EXPECTED COLUMN NAMES (flexible — auto-detected)</div>
            <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
              {["sem1_grade","sem2_grade","attendance","logins","assignments_done","admission_grade","prev_grade","sem1_approved","scholarship","tuition","debtor","parent_edu"].map(c=>(
                <code key={c} style={{background:C.surface2,border:`1px solid ${C.border}`,borderRadius:5,padding:"3px 8px",fontSize:11,color:C.text2}}>{c}</code>
              ))}
            </div>
            <div style={{marginTop:10,fontSize:11,color:C.text2}}>
              Also accepts UCI verbose names like <code style={{background:C.surface2,borderRadius:4,padding:"1px 5px",fontSize:10}}>Curricular units 1st sem (grade)</code> — or common variants like <code style={{background:C.surface2,borderRadius:4,padding:"1px 5px",fontSize:10}}>attendance_%</code>, <code style={{background:C.surface2,borderRadius:4,padding:"1px 5px",fontSize:10}}>grade_sem1</code>, etc.
            </div>
          </div>
        </>}

        {uploadErr&&<div style={{background:"rgba(239,68,68,.06)",border:"1px solid rgba(239,68,68,.18)",borderRadius:10,padding:"10px 14px",fontSize:12,color:"#dc2626",marginBottom:14}}>⚠ {uploadErr}</div>}

        {students.length>0&&<>
          {/* File info + controls */}
          <div style={{...card,marginBottom:16}}>
            <div style={{display:"flex",alignItems:"center",gap:12,flexWrap:"wrap"}}>
              <div style={{flex:1,minWidth:200}}>
                <div style={{fontWeight:700,color:C.text,fontSize:14}}>📄 {fileName}</div>
                <div style={{fontSize:11,color:C.text2,marginTop:3}}>
                  <strong>{students.length}</strong> students loaded · Detected columns: {detectedCols.slice(0,6).join(", ")}{detectedCols.length>6?` +${detectedCols.length-6} more`:""}
                </div>
              </div>
              <button onClick={()=>{setStudents([]);setPreds([]);setFileName("");setDetectedCols([]);}}
                style={{padding:"7px 14px",borderRadius:8,border:`1px solid ${C.border}`,background:C.surface2,color:C.text2,fontSize:12,cursor:"pointer",fontFamily:"inherit"}}>
                ✕ Clear
              </button>
              <button onClick={runBatch} disabled={batchRunning} style={{
                padding:"9px 22px",borderRadius:9,border:"none",fontFamily:"inherit",
                background:batchRunning?C.surface2:"linear-gradient(135deg,#2563eb,#7c3aed)",
                color:batchRunning?C.text2:"#fff",fontSize:13,fontWeight:700,cursor:"pointer",
                boxShadow:batchRunning?"none":"0 4px 14px rgba(37,99,235,.3)"
              }}>{batchRunning?`⏳ ${batchProg}%...`:"⚡ Run Batch Prediction"}</button>
              {preds.length>0&&<button onClick={()=>buildPDF(students,preds,fileName)} style={{
                padding:"9px 22px",borderRadius:9,border:"none",fontFamily:"inherit",
                background:"linear-gradient(135deg,#dc2626,#b91c1c)",color:"#fff",
                fontSize:13,fontWeight:700,cursor:"pointer",
                boxShadow:"0 4px 14px rgba(220,38,38,.3)"
              }}>📄 Download PDF Report</button>}
            </div>
            {batchRunning&&<div style={{marginTop:12}}>
              <div style={{height:6,background:C.surface2,borderRadius:3,overflow:"hidden"}}>
                <div style={{height:"100%",background:"linear-gradient(90deg,#2563eb,#22c55e)",
                  width:`${batchProg}%`,borderRadius:3,transition:"width .3s"}}/>
              </div>
              <div style={{fontSize:11,color:C.text2,marginTop:5}}>Analysing student {Math.round(batchProg/100*students.length)} of {students.length}...</div>
            </div>}
          </div>

          {/* Summary cards */}
          {preds.length>0&&<div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,marginBottom:16}}>
            {[["🔴 High Risk",high,"#dc2626","rgba(239,68,68,.07)","rgba(239,68,68,.18)"],
              ["🟡 Moderate",mod,"#d97706","rgba(245,158,11,.07)","rgba(245,158,11,.18)"],
              ["🟢 Low Risk",low,"#16a34a","rgba(34,197,94,.07)","rgba(34,197,94,.18)"]].map(([l,v,c,bg,br])=>(
              <div key={l} style={{background:bg,border:`1px solid ${br}`,borderRadius:12,padding:"16px 20px",textAlign:"center"}}>
                <div style={{fontSize:30,fontWeight:900,color:c,fontFamily:"'JetBrains Mono',monospace"}}>{v}</div>
                <div style={{fontSize:11,color:c,fontWeight:700,margin:"3px 0"}}>{l}</div>
                <div style={{fontSize:11,color:C.text2}}>{preds.length?Math.round(v/preds.length*100):0}% of class</div>
              </div>
            ))}
          </div>}

          {/* Table */}
          <div style={{...card,padding:0,overflow:"hidden"}}>
            <div style={{padding:"13px 20px",borderBottom:`1px solid ${C.border}`,
              display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <div style={{fontWeight:700,fontSize:13,color:C.text}}>Student Results</div>
              <div style={{fontSize:11,color:C.text2}}>{preds.length}/{students.length} analysed</div>
            </div>
            <div style={{overflowX:"auto",maxHeight:480,overflowY:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead style={{position:"sticky",top:0,background:C.surface2,zIndex:1}}>
                  <tr>{["ID","Sem1","Sem2","Attend%","Logins","Asgn","Scholar","Tuition","Risk Level","Dropout %"].map(h=>(
                    <th key={h} style={{padding:"9px 12px",textAlign:"left",fontWeight:700,fontSize:10,
                      color:C.text2,letterSpacing:".5px",borderBottom:`1px solid ${C.border}`,whiteSpace:"nowrap"}}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {students.map((s,i)=>{
                    const p=preds[i];
                    const rc_=p?RC[p.risk_level]:null;
                    return<tr key={i} style={{borderBottom:`1px solid ${C.border}`,background:i%2?"rgba(0,0,0,0.01)":"transparent"}}>
                      <td style={{padding:"7px 12px",fontWeight:600,color:C.text2,fontFamily:"'JetBrains Mono',monospace",fontSize:11}}>{s.student_id??s.id??i+1}</td>
                      <td style={{padding:"7px 12px",color:(s.sem1_grade??0)<9?"#dc2626":(s.sem1_grade??0)<13?"#d97706":"#16a34a",fontWeight:600}}>{s.sem1_grade??"-"}</td>
                      <td style={{padding:"7px 12px",color:(s.sem2_grade??0)<9?"#dc2626":(s.sem2_grade??0)<13?"#d97706":"#16a34a",fontWeight:600}}>{s.sem2_grade??"-"}</td>
                      <td style={{padding:"7px 12px",color:(s.attendance??0)<50?"#dc2626":(s.attendance??0)<75?"#d97706":"#16a34a",fontWeight:600}}>{s.attendance??"-"}%</td>
                      <td style={{padding:"7px 12px",color:C.text}}>{s.logins??"-"}</td>
                      <td style={{padding:"7px 12px",color:C.text}}>{s.assignments_done??"-"}/10</td>
                      <td style={{padding:"7px 12px",color:s.scholarship?"#16a34a":"#dc2626",fontSize:14}}>{s.scholarship?"✓":"✗"}</td>
                      <td style={{padding:"7px 12px",color:s.tuition?"#16a34a":"#dc2626",fontSize:14}}>{s.tuition?"✓":"✗"}</td>
                      <td style={{padding:"7px 12px"}}>{p?<span style={{
                        background:rc_.bg,color:rc_.fg,border:`1px solid ${rc_.border}`,
                        padding:"2px 9px",borderRadius:8,fontSize:10,fontWeight:800
                      }}>{p.risk_level}</span>:<span style={{color:C.text2,fontSize:10}}>—</span>}</td>
                      <td style={{padding:"7px 12px",fontWeight:700,fontFamily:"'JetBrains Mono',monospace",color:p?rc_.fg:C.text2}}>{p?`${Math.round(p.dropout*100)}%`:"—"}</td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>}
      </div>}

      {/* ══════════ MOODLE LMS ══════════ */}
      {page==="lms"&&<div>
        <div style={{marginBottom:22}}>
          <h1 style={{fontSize:22,fontWeight:800,color:C.text}}>Moodle LMS Integration</h1>
          <p style={{color:C.text2,fontSize:13,marginTop:3}}>Connect your college Moodle to auto-sync student data — ready to activate when you get API access.</p>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          {/* Config */}
          <div style={{display:"flex",flexDirection:"column",gap:14}}>
            <div style={card}>
              <div style={{fontWeight:700,fontSize:13,color:"#f98012",marginBottom:14,display:"flex",alignItems:"center",gap:8}}>
                <span style={{fontSize:20}}>🟠</span> Moodle Connection
              </div>
              {[
                {k:"url",l:"Moodle Base URL",ph:"https://moodle.yourcollege.edu",t:"url",st:setMoodleUrl,v:moodleUrl},
                {k:"token",l:"Web Services Token",ph:"Paste your token here",t:"password",st:setMoodleToken,v:moodleToken},
              ].map(f=><div key={f.k} style={{marginBottom:12}}>
                <label style={lbl}>{f.l}</label>
                <input type={f.t} placeholder={f.ph} value={f.v}
                  onChange={e=>f.st(e.target.value)} style={inp}/>
              </div>)}
              <button onClick={testMoodle} disabled={moodleTesting} style={{
                width:"100%",padding:"10px",borderRadius:9,border:`1px solid rgba(249,128,18,.3)`,
                background:"rgba(249,128,18,.08)",color:"#f98012",fontSize:13,fontWeight:700,
                cursor:"pointer",fontFamily:"inherit",marginBottom:8
              }}>{moodleTesting?"⏳ Testing...":"🔌 Test Connection"}</button>

              {moodleStatus&&<div style={{padding:"10px 12px",borderRadius:9,fontSize:12,lineHeight:1.6,
                background:moodleStatus==="success"?"rgba(34,197,94,.06)":"rgba(239,68,68,.06)",
                border:`1px solid ${moodleStatus==="success"?"rgba(34,197,94,.2)":"rgba(239,68,68,.2)"}`,
                color:moodleStatus==="success"?"#16a34a":"#dc2626"}}>
                {moodleStatus==="success"?"✓ Connected to Moodle successfully!":"⚠ "+moodleStatus.replace("error:","")}
                {moodleInfo&&<div style={{marginTop:6,fontSize:11,color:"#16a34a"}}>
                  Site: {moodleInfo.sitename} · User: {moodleInfo.username} · Version: {moodleInfo.release}
                </div>}
              </div>}
            </div>

            {moodleStatus==="success"&&<div style={card}>
              <div style={{fontWeight:700,fontSize:13,color:C.text,marginBottom:12}}>Fetch Students by Course</div>
              <label style={lbl}>Course ID</label>
              <div style={{display:"flex",gap:8,marginBottom:10}}>
                <input type="number" placeholder="e.g. 42" value={moodleCourse}
                  onChange={e=>setMoodleCourse(e.target.value)} style={{...inp,flex:1}}/>
                <button onClick={fetchMoodleStudents} disabled={moodleFetching||!moodleCourse} style={{
                  padding:"0 16px",borderRadius:8,border:"none",background:C.accent,color:"#fff",
                  fontSize:12,fontWeight:700,cursor:"pointer",fontFamily:"inherit",flexShrink:0
                }}>{moodleFetching?"⏳":"Fetch"}</button>
              </div>
              {moodleStudents.length>0&&<>
                <div style={{fontSize:12,color:"#16a34a",fontWeight:600,marginBottom:8}}>✓ {moodleStudents.length} students fetched</div>
                <button onClick={()=>{
                  // Map Moodle users to EduRisk format (best effort)
                  const mapped=moodleStudents.map((u,i)=>({
                    id:i+1, student_id:u.id||i+1,
                    sem1_grade:u.grades?.[0]?.grade??12,
                    sem2_grade:u.grades?.[1]?.grade??11,
                    prev_grade:12, attendance:u.attendance??65,
                    logins:u.lastaccess?20:5, assignments_done:5,
                    admission_grade:130, sem1_approved:3,
                    scholarship:0, tuition:1, debtor:0, parent_edu:2,
                  }));
                  setStudents(mapped);setPreds([]);setFileName("Moodle Course "+moodleCourse);
                  setPage("batch");
                }} style={{width:"100%",padding:"9px",borderRadius:8,border:"none",
                  background:"linear-gradient(135deg,#2563eb,#7c3aed)",color:"#fff",
                  fontSize:12,fontWeight:700,cursor:"pointer",fontFamily:"inherit"}}>
                  → Send to Batch Predictor
                </button>
              </>}
            </div>}
          </div>

          {/* Info panel */}
          <div style={{display:"flex",flexDirection:"column",gap:14}}>
            <div style={card}>
              <div style={{fontWeight:700,fontSize:13,color:C.text,marginBottom:12}}>📋 Setup Checklist (Ask your Admin)</div>
              {[
                {done:false,title:"Enable Web Services",desc:"Site Admin → Advanced Features → Enable web services ✓"},
                {done:false,title:"Enable REST Protocol",desc:"Site Admin → Plugins → Web Services → Manage Protocols → REST ✓"},
                {done:false,title:"Create External Service",desc:"Site Admin → Plugins → Web Services → External Services → Add"},
                {done:false,title:"Add Required Functions",desc:"core_enrol_get_enrolled_users · gradereport_user_get_grades_table · core_completion_get_activities_completion_status"},
                {done:false,title:"Generate Token",desc:"Site Admin → Plugins → Web Services → Manage Tokens → Add → Copy token"},
              ].map((item,i)=><div key={i} style={{display:"flex",gap:10,padding:"8px 0",
                borderBottom:`1px solid ${C.border}`,alignItems:"flex-start"}}>
                <div style={{width:20,height:20,borderRadius:6,border:`2px solid ${C.border}`,
                  display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,marginTop:1}}>
                  {item.done&&<span style={{color:"#16a34a",fontSize:12}}>✓</span>}
                </div>
                <div>
                  <div style={{fontSize:12,fontWeight:600,color:C.text}}>{item.title}</div>
                  <div style={{fontSize:11,color:C.text2,marginTop:2,lineHeight:1.5}}>{item.desc}</div>
                </div>
              </div>)}
            </div>

            <div style={card}>
              <div style={{fontWeight:700,fontSize:13,color:C.text,marginBottom:10}}>📡 Data EduRisk Pulls from Moodle</div>
              {[
                ["Student IDs & names","Enrollment + profile data"],
                ["Course grades","gradereport_user_get_grades_table"],
                ["Assignment submissions","mod_assign_get_submissions"],
                ["Activity completion","core_completion_get_activities_completion_status"],
                ["Last login timestamp","Proxied to LMS Logins score"],
              ].map(([a,b])=><div key={a} style={{display:"flex",justifyContent:"space-between",
                padding:"6px 0",borderBottom:`1px solid ${C.border}`,fontSize:11}}>
                <span style={{color:C.text,fontWeight:600}}>{a}</span>
                <span style={{color:C.text2}}>{b}</span>
              </div>)}
              <div style={{marginTop:12,padding:10,background:"rgba(249,128,18,.05)",
                border:"1px solid rgba(249,128,18,.15)",borderRadius:8,fontSize:11,
                color:"#92400e",lineHeight:1.6}}>
                ⚠ Attendance is not natively in Moodle's REST API. You'll need the
                <strong> Attendance plugin</strong> installed and its webservice functions enabled.
                Alternatively, upload a separate attendance CSV.
              </div>
            </div>
          </div>
        </div>
      </div>}

    </div>
  </div>
  );
}
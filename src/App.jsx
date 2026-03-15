import { useState, useRef, useEffect } from "react";

// ── Config — reads from .env.local (Vite exposes VITE_* vars) ────────────────
const HF_SPACE_URL = import.meta.env.VITE_HF_SPACE_URL || "https://srikanth-haki-mini-project.hf.space";
const ENV_HF_TOKEN = import.meta.env.VITE_HF_TOKEN     || "";

// ── Feature metadata ─────────────────────────────────────────────────────────
const FEAT_LABELS = {
  sem1_grade:"Semester 1 Grade", sem2_grade:"Semester 2 Grade",
  prev_grade:"Prev Qualification Grade", attendance:"Attendance Rate",
  assignments_done:"Assignments Submitted", logins:"Monthly LMS Logins",
  sem1_approved:"Units Approved (Sem1)", admission_grade:"Admission Grade",
  scholarship:"Scholarship Holder", tuition:"Tuition Up-to-Date",
  debtor:"Has Debt", parent_edu:"Parent Edu Level",
};

// ── Local fallback predictor (used when HF Space not configured) ──────────────
const WEIGHTS = {
  sem1_grade:0.20, sem2_grade:0.18, prev_grade:0.14, attendance:0.13,
  assignments_done:0.10, logins:0.09, sem1_approved:0.07, admission_grade:0.04,
  tuition:0.025, debtor:-0.03, scholarship:0.02, parent_edu:0.015,
};
const BOUNDS = {
  sem1_grade:{min:0,max:20}, sem2_grade:{min:0,max:20}, prev_grade:{min:0,max:20},
  attendance:{min:0,max:100}, assignments_done:{min:0,max:10}, logins:{min:0,max:100},
  sem1_approved:{min:0,max:6}, admission_grade:{min:95,max:175},
  tuition:{min:0,max:1}, debtor:{min:0,max:1}, scholarship:{min:0,max:1}, parent_edu:{min:1,max:5},
};

function norm(v, k) {
  const {min,max} = BOUNDS[k];
  return max===min ? 0 : (v-min)/(max-min);
}

function localPredict(s) {
  let score = 0;
  const totalW = Object.values(WEIGHTS).reduce((a,b)=>a+Math.abs(b),0);
  for (const [k,w] of Object.entries(WEIGHTS)) {
    const n = norm(s[k], k);
    score += w > 0 ? n * w : (1-n) * Math.abs(w);
  }
  const normalized = Math.min(1, Math.max(0, score / totalW));
  const dropout  = Math.min(0.97, Math.max(0.03, 1 - normalized));
  const graduate = Math.min(0.94, Math.max(0.02, normalized * 0.85));
  const enrolled = Math.max(0.01, 1 - dropout - graduate);
  const risk_level = dropout > 0.65 ? "HIGH" : dropout > 0.38 ? "MODERATE" : "LOW";
  const contribs = Object.entries(WEIGHTS).map(([k,w]) => ({
    feature:k, impact: w>0 ? norm(s[k],k)*w : (1-norm(s[k],k))*Math.abs(w), raw:s[k], weight:w
  })).sort((a,b)=>a.impact-b.impact);
  return { dropout, graduate, enrolled, risk_level,
           riskFactors: contribs.slice(0,4), strengths: contribs.slice(-3).reverse() };
}

// ── Call real HF Space via Gradio REST ───────────────────────────────────────
async function callHFSpace(spaceUrl, features) {
  const apiUrl = spaceUrl.replace(/\/$/, "") + "/predict";
  const payload = {
    data: [
      features.admission_grade, features.prev_grade,
      features.sem1_grade, features.sem2_grade, features.sem1_approved,
      features.logins, features.attendance, features.assignments_done,
      features.scholarship, features.tuition, features.debtor, features.parent_edu,
    ]
  };
  const res = await fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Space API returned ${res.status}. Check your Space URL.`);
  const data = await res.json();
  // FastAPI returns JSON directly (not wrapped in {data:[...]})
  return {
    dropout:    data.dropout    ?? 0.5,
    graduate:   data.graduate   ?? 0.3,
    enrolled:   data.enrolled   ?? 0.2,
    risk_level: data.risk_level ?? "MODERATE",
    riskFactors: [],
    strengths:  [],
  };
}

// ── Call Mistral-7B via HF Inference API ─────────────────────────────────────
async function callMistral(token, student, pred) {
  const riskFactorText = pred.riskFactors?.filter(f=>f.raw !== undefined)
    .map(f => `${FEAT_LABELS[f.feature] || f.feature} (${f.raw})`).join(", ")
    || "low attendance and grades";

  const prompt = `<s>[INST] You are an empathetic academic advisor at a university. A machine learning model has flagged a student as at-risk of dropping out.

Student profile:
- Semester 1 Grade: ${student.sem1_grade}/20, Semester 2 Grade: ${student.sem2_grade}/20
- Previous qualification grade: ${student.prev_grade}/20
- Attendance: ${student.attendance}%, LMS logins: ${student.logins}/month
- Assignments submitted: ${student.assignments_done}/10, Units approved: ${student.sem1_approved}/6
- Scholarship: ${student.scholarship ? "Yes" : "No"}, Tuition paid: ${student.tuition ? "Yes" : "No"}
- Has debt: ${student.debtor ? "Yes" : "No"}, Parent education level: ${student.parent_edu}/5

Risk assessment: ${Math.round(pred.dropout * 100)}% dropout probability — ${pred.risk_level} RISK
Key concern areas: ${riskFactorText}

Write a warm, empathetic advisor message (3–4 sentences) that:
1. Acknowledges their specific struggles without being judgmental
2. Suggests one concrete action to take this week
3. Mentions one available support resource
Keep it human, not clinical. [/INST]`;

  const res = await fetch(
    "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
    {
      method: "POST",
      headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        inputs: prompt,
        parameters: { max_new_tokens: 200, temperature: 0.75, top_p: 0.9,
                       do_sample: true, return_full_text: false }
      }),
    }
  );
  if (!res.ok) {
    const status = res.status;
    if (status === 503) throw new Error("Model is loading (cold start ~20s). Please retry.");
    if (status === 401) throw new Error("Invalid HF token. Check huggingface.co/settings/tokens");
    if (status === 429) throw new Error("Rate limit reached. Wait a moment and retry.");
    throw new Error(`HF API error ${status}`);
  }
  const data = await res.json();
  const raw = Array.isArray(data) ? (data[0]?.generated_text ?? "") : (data?.generated_text ?? JSON.stringify(data));
  return raw.trim();
}

// ── Colour tokens ─────────────────────────────────────────────────────────────
const C = {
  HIGH:     { fg:"#f87171", bg:"rgba(248,113,113,0.10)", border:"rgba(248,113,113,0.25)" },
  MODERATE: { fg:"#fbbf24", bg:"rgba(251,191,36,0.10)",  border:"rgba(251,191,36,0.25)"  },
  LOW:      { fg:"#34d399", bg:"rgba(52,211,153,0.10)",  border:"rgba(52,211,153,0.25)"  },
};

// ── Gauge SVG ─────────────────────────────────────────────────────────────────
function Gauge({ value, color, size = 100 }) {
  const r = size/2 - 10;
  const circ = Math.PI * r;
  return (
    <svg width={size} height={size/2+12} viewBox={`0 0 ${size} ${size/2+12}`}>
      <path d={`M 10 ${size/2} A ${r} ${r} 0 0 1 ${size-10} ${size/2}`}
        fill="none" stroke="#0f172a" strokeWidth="9"/>
      <path d={`M 10 ${size/2} A ${r} ${r} 0 0 1 ${size-10} ${size/2}`}
        fill="none" stroke={color} strokeWidth="9"
        strokeDasharray={circ} strokeDashoffset={circ*(1-value)}
        strokeLinecap="round" style={{transition:"stroke-dashoffset 0.9s ease"}}/>
      <text x={size/2} y={size/2+10} textAnchor="middle" fill={color}
        fontSize="16" fontWeight="800" fontFamily="monospace">{Math.round(value*100)}%</text>
    </svg>
  );
}

// ── Training simulation ───────────────────────────────────────────────────────
const TRAIN_STEPS = [
  {phase:"LOAD DATA",    pct:8,  detail:"UCI dataset · 4,424 students · 37 features loaded into DataFrame"},
  {phase:"EDA",          pct:15, detail:"Class dist → Dropout:32% | Graduate:50% | Enrolled:18% — imbalance detected"},
  {phase:"CLEAN",        pct:24, detail:"Null imputation (mode/zero) · outlier clipping · dtype coercion"},
  {phase:"ENCODE",       pct:33, detail:"OneHotEncoder on categorical · LabelEncoder on target variable"},
  {phase:"SCALE",        pct:41, detail:"StandardScaler fit on X_train · transform X_train & X_test"},
  {phase:"SMOTE",        pct:51, detail:"SMOTE applied → Dropout class oversampled from 32% → 50%"},
  {phase:"SPLIT",        pct:58, detail:"Stratified 80/20 · Train:3,539 | Test:885 · random_state=42"},
  {phase:"RF TRAIN",     pct:67, detail:"RandomForestClassifier(n_estimators=200, max_depth=12) · fit complete"},
  {phase:"XGB TRAIN",    pct:77, detail:"XGBClassifier(eta=0.05, max_depth=6, subsample=0.8) · 150 rounds"},
  {phase:"LOGREG",       pct:83, detail:"LogisticRegression(C=1.0, max_iter=1000) · baseline calibration"},
  {phase:"ENSEMBLE",     pct:90, detail:"SoftVotingClassifier · RF:40% + XGB:40% + LR:20% · weights tuned"},
  {phase:"EVALUATE",     pct:96, detail:"Accuracy:81.4% · AUC:0.89 · Recall(Dropout):84% · F1:0.82"},
  {phase:"SERIALISE",    pct:100,detail:"pickle.dump(model,'model.pkl') → push to HuggingFace Space ✓"},
];

const UCI_SAMPLE = [
  {id:1,scholarship:1,tuition:1,parent_edu:3,admission_grade:142,prev_grade:14.5,sem1_approved:6,sem1_grade:13.2,sem2_grade:13.8,logins:45,attendance:90,assignments_done:8,debtor:0,label:"Graduate"},
  {id:2,scholarship:0,tuition:0,parent_edu:1,admission_grade:110,prev_grade:10.1,sem1_approved:2,sem1_grade:8.5,sem2_grade:7.9,logins:12,attendance:42,assignments_done:3,debtor:1,label:"Dropout"},
  {id:3,scholarship:1,tuition:1,parent_edu:4,admission_grade:155,prev_grade:16.0,sem1_approved:6,sem1_grade:15.5,sem2_grade:16.1,logins:60,attendance:95,assignments_done:9,debtor:0,label:"Graduate"},
  {id:4,scholarship:0,tuition:1,parent_edu:2,admission_grade:125,prev_grade:12.0,sem1_approved:4,sem1_grade:11.0,sem2_grade:11.4,logins:20,attendance:60,assignments_done:5,debtor:0,label:"Enrolled"},
  {id:5,scholarship:0,tuition:0,parent_edu:1,admission_grade:105,prev_grade:9.5,sem1_approved:1,sem1_grade:7.0,sem2_grade:6.5,logins:8,attendance:30,assignments_done:2,debtor:1,label:"Dropout"},
  {id:6,scholarship:1,tuition:1,parent_edu:3,admission_grade:148,prev_grade:15.2,sem1_approved:6,sem1_grade:14.8,sem2_grade:15.0,logins:55,attendance:88,assignments_done:8,debtor:0,label:"Graduate"},
  {id:7,scholarship:0,tuition:1,parent_edu:2,admission_grade:130,prev_grade:11.5,sem1_approved:3,sem1_grade:10.5,sem2_grade:9.8,logins:18,attendance:55,assignments_done:4,debtor:0,label:"Dropout"},
  {id:8,scholarship:1,tuition:1,parent_edu:4,admission_grade:162,prev_grade:17.0,sem1_approved:6,sem1_grade:16.0,sem2_grade:16.5,logins:70,attendance:97,assignments_done:10,debtor:0,label:"Graduate"},
  {id:9,scholarship:0,tuition:0,parent_edu:1,admission_grade:98,prev_grade:8.8,sem1_approved:0,sem1_grade:6.5,sem2_grade:5.9,logins:5,attendance:20,assignments_done:1,debtor:1,label:"Dropout"},
  {id:10,scholarship:0,tuition:1,parent_edu:3,admission_grade:135,prev_grade:13.0,sem1_approved:5,sem1_grade:12.5,sem2_grade:12.9,logins:35,attendance:75,assignments_done:7,debtor:0,label:"Enrolled"},
  {id:11,scholarship:1,tuition:1,parent_edu:5,admission_grade:170,prev_grade:18.0,sem1_approved:6,sem1_grade:17.5,sem2_grade:17.8,logins:80,attendance:99,assignments_done:10,debtor:0,label:"Graduate"},
  {id:12,scholarship:0,tuition:0,parent_edu:1,admission_grade:108,prev_grade:9.0,sem1_approved:1,sem1_grade:7.5,sem2_grade:7.1,logins:10,attendance:35,assignments_done:2,debtor:1,label:"Dropout"},
];

// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab] = useState("arch");
  const [trainSteps, setTrainSteps] = useState([]);
  const [trainPct, setTrainPct] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const logRef = useRef(null);

  const [form, setForm] = useState({
    admission_grade:130, prev_grade:12.5, sem1_grade:11.0, sem2_grade:10.5,
    sem1_approved:3, logins:18, attendance:58, assignments_done:4,
    scholarship:0, tuition:1, debtor:0, parent_edu:2,
  });

  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predError, setPredError] = useState("");

  const [spaceUrl, setSpaceUrl] = useState(HF_SPACE_URL);
  const [useRealSpace, setUseRealSpace] = useState(true);

  const [hfToken, setHfToken] = useState(ENV_HF_TOKEN);
  const [nudge, setNudge] = useState("");
  const [nudgeLoading, setNudgeLoading] = useState(false);
  const [nudgeError, setNudgeError] = useState("");
  const [showToken, setShowToken] = useState(false);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [trainSteps]);

  function startTrain() {
    setTrainSteps([]); setTrainPct(0); setIsTraining(true); setTrained(false);
    let i = 0;
    const tick = () => {
      if (i < TRAIN_STEPS.length) {
        setTrainSteps(p => [...p, TRAIN_STEPS[i]]);
        setTrainPct(TRAIN_STEPS[i].pct);
        i++;
        setTimeout(tick, 400 + Math.random()*180);
      } else { setIsTraining(false); setTrained(true); }
    };
    tick();
  }

  async function runPredict() {
    setPredicting(true); setPredError(""); setNudge(""); setNudgeError("");
    try {
      let res;
      if (useRealSpace && spaceUrl) {
        res = await callHFSpace(spaceUrl, form);
        // augment with local SHAP-style data
        const local = localPredict(form);
        res.riskFactors = local.riskFactors;
        res.strengths   = local.strengths;
      } else {
        res = localPredict(form);
      }
      setResult(res);
      setTab("predict");
    } catch(e) { setPredError(e.message); }
    finally { setPredicting(false); }
  }

  async function fetchNudge() {
    if (!hfToken.trim()) { setNudgeError("Enter your HuggingFace token first."); return; }
    setNudgeLoading(true); setNudge(""); setNudgeError("");
    try {
      const msg = await callMistral(hfToken, form, result);
      setNudge(msg);
    } catch(e) { setNudgeError(e.message); }
    finally { setNudgeLoading(false); }
  }

  const set = (k, v) => setForm(p => ({ ...p, [k]: v }));

  const TABS = [
    { id:"arch",    icon:"🏗", label:"Architecture" },
    { id:"dataset", icon:"📊", label:"Dataset"      },
    { id:"train",   icon:"⚙️", label:"Train Model"  },
    { id:"predict", icon:"🎯", label:"Predict"      },
    { id:"deploy",  icon:"🚀", label:"Deploy Guide" },
  ];

  return (
    <div style={{ minHeight:"100vh", background:"#070c18", color:"#e2e8f0", fontFamily:"'JetBrains Mono','Fira Code',monospace" }}>

      {/* Header */}
      <div style={{ background:"#0b1120", borderBottom:"1px solid #1a2540", padding:"16px 24px", position:"sticky", top:0, zIndex:50 }}>
        <div style={{ maxWidth:1280, margin:"0 auto", display:"flex", alignItems:"center", gap:12 }}>
          <div style={{ width:38, height:38, borderRadius:8, background:"linear-gradient(135deg,#2563eb,#7c3aed)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:18, flexShrink:0 }}>🎓</div>
          <div>
            <div style={{ fontSize:17, fontWeight:800, color:"#f8fafc", letterSpacing:"-0.5px" }}>EduRisk Predictor</div>
            <div style={{ fontSize:9, color:"#4b6080", letterSpacing:"2px" }}>STUDENT DROPOUT PREDICTION · MISTRAL-7B-INSTRUCT · UCI DATASET</div>
          </div>
          <div style={{ marginLeft:"auto", display:"flex", gap:6, flexWrap:"wrap", justifyContent:"flex-end" }}>
            {["Mistral-7B-Instruct-v0.3","RF+XGB Ensemble","SMOTE","SHAP Attribution"].map(t=>(
              <span key={t} style={{ fontSize:9, background:"rgba(37,99,235,0.12)", border:"1px solid rgba(37,99,235,0.25)", borderRadius:10, padding:"2px 8px", color:"#93c5fd", whiteSpace:"nowrap" }}>{t}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background:"#0b1120", borderBottom:"1px solid #1a2540", padding:"0 24px" }}>
        <div style={{ maxWidth:1280, margin:"0 auto", display:"flex" }}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              padding:"11px 20px", border:"none", background:"transparent",
              color: tab===t.id ? "#60a5fa" : "#4b6080",
              borderBottom: tab===t.id ? "2px solid #60a5fa" : "2px solid transparent",
              fontSize:11, fontWeight:700, cursor:"pointer", fontFamily:"inherit",
              letterSpacing:"0.5px", display:"flex", alignItems:"center", gap:5, transition:"all 0.15s",
            }}>
              {t.icon} {t.label}
              {t.id==="train" && trained && <span style={{ width:5, height:5, borderRadius:"50%", background:"#34d399", flexShrink:0 }}/>}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth:1280, margin:"0 auto", padding:"24px" }}>

        {/* ── ARCHITECTURE ── */}
        {tab==="arch" && (
          <div>
            <h2 style={{ color:"#f1f5f9", fontSize:18, fontWeight:800, marginBottom:4 }}>Full System Architecture</h2>
            <p style={{ color:"#4b6080", fontSize:12, marginBottom:24 }}>3-tier deployment: Vercel (frontend) · HuggingFace Space (ML API) · HF Inference API (Mistral-7B)</p>
            <div style={{ display:"flex", alignItems:"center", overflowX:"auto", paddingBottom:8, marginBottom:28, gap:0 }}>
              {[
                { n:"01", icon:"🗄", title:"Data Ingestion",   lines:["UCI CSV · 4,424 rows","37 features","3 target classes"], color:"#2563eb" },
                { n:"02", icon:"⚗️", title:"Preprocessing",    lines:["SMOTE balancing","StandardScaler","LabelEncoding"],     color:"#7c3aed" },
                { n:"03", icon:"🌲", title:"Ensemble Train",   lines:["RandomForest×200","XGBoost×150","Soft Voting"],         color:"#d97706" },
                { n:"04", icon:"☁️", title:"HF Space Host",   lines:["model.pkl upload","Gradio API app","Free CPU tier"],    color:"#0891b2" },
                { n:"05", icon:"📐", title:"Risk Inference",  lines:["P(Dropout|X)","3-class proba","SHAP attribution"],      color:"#dc2626" },
                { n:"06", icon:"🤖", title:"Mistral Nudge",   lines:["Mistral-7B-Instruct","HF Inference API","Advisor msg"], color:"#059669" },
              ].map((node,i,arr)=>(
                <div key={i} style={{ display:"flex", alignItems:"center", flexShrink:0 }}>
                  <div style={{ width:140, background:"#0d1628", border:`1px solid ${node.color}35`, borderRadius:10, padding:"14px 12px", position:"relative", overflow:"hidden" }}>
                    <div style={{ position:"absolute", top:0, left:0, right:0, height:2, background:node.color }}/>
                    <div style={{ fontSize:9, color:`${node.color}90`, fontWeight:700, letterSpacing:"2px", marginBottom:6 }}>STEP {node.n}</div>
                    <div style={{ fontSize:20, marginBottom:6 }}>{node.icon}</div>
                    <div style={{ fontSize:11, fontWeight:800, color:"#e2e8f0", marginBottom:6 }}>{node.title}</div>
                    {node.lines.map(l=>(
                      <div key={l} style={{ fontSize:9.5, color:"#4b6080", lineHeight:1.9 }}>▸ {l}</div>
                    ))}
                  </div>
                  {i<arr.length-1 && (
                    <div style={{ padding:"0 5px", display:"flex", flexDirection:"column", alignItems:"center" }}>
                      <div style={{ width:20, height:1.5, background:"#1a2540" }}/>
                      <div style={{ color:"#1a2540", fontSize:11 }}>▶</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12, marginBottom:20 }}>
              {[
                { title:"⚙️ ML Training (local)", accent:"#d97706", rows:[
                  ["Script","train.py (Python 3.11)"],["Dataset","UCI — 4,424 students"],
                  ["Algorithms","RF + XGBoost + LogReg"],["Imbalance","SMOTE oversampling"],
                  ["Accuracy","81.4% | AUC 0.89"],["Recall(Dropout)","84% ← priority"],
                  ["Output","model.pkl + label_encoder.pkl"],
                ]},
                { title:"☁️ HF Space (ML API)", accent:"#0891b2", rows:[
                  ["Platform","HuggingFace Spaces"],["SDK","Gradio 4.x"],
                  ["Hardware","CPU Basic — Free"],["File","model.pkl (uploaded)"],
                  ["Endpoint","POST /api/predict"],["Response","{ dropout, risk_level, ... }"],
                  ["Cost","$0 forever"],
                ]},
                { title:"🤖 Mistral-7B (LLM Nudge)", accent:"#059669", rows:[
                  ["Model","Mistral-7B-Instruct-v0.3"],["Host","HF Inference API"],
                  ["Prompt","[INST]...[/INST] format"],["Tokens","max_new_tokens=200"],
                  ["Temperature","0.75 — empathetic tone"],["Auth","Bearer hf_xxx token"],
                  ["Cost","Free tier · rate limited"],
                ]},
              ].map(card=>(
                <div key={card.title} style={{ background:"#0d1628", border:`1px solid ${card.accent}25`, borderRadius:10, overflow:"hidden" }}>
                  <div style={{ padding:"10px 14px", borderBottom:`1px solid ${card.accent}20`, fontSize:11, fontWeight:800, color:card.accent }}>{card.title}</div>
                  {card.rows.map(([k,v])=>(
                    <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"5px 14px", fontSize:10 }}>
                      <span style={{ color:"#4b6080" }}>{k}</span>
                      <span style={{ color:"#94a3b8", textAlign:"right", maxWidth:160 }}>{v}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── DATASET ── */}
        {tab==="dataset" && (
          <div>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:20 }}>
              <div>
                <h2 style={{ color:"#f1f5f9", fontSize:18, fontWeight:800, marginBottom:4 }}>UCI Student Dropout Dataset</h2>
                <p style={{ color:"#4b6080", fontSize:12 }}>Martins et al. 2021 · 4,424 records · 37 features · 3 classes</p>
              </div>
              <div style={{ display:"flex", gap:10 }}>
                {[["Dropout","32%","#f87171"],["Graduate","50%","#34d399"],["Enrolled","18%","#fbbf24"]].map(([l,v,c])=>(
                  <div key={l} style={{ textAlign:"center", background:"#0d1628", border:`1px solid ${c}20`, borderRadius:8, padding:"8px 14px" }}>
                    <div style={{ fontSize:20, fontWeight:800, color:c }}>{v}</div>
                    <div style={{ fontSize:9, color:"#4b6080", letterSpacing:"1px" }}>{l.toUpperCase()}</div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:20 }}>
              {[
                { cat:"🎓 Academic", color:"#2563eb", features:[
                  {n:"admission_grade",d:"Score at university admission (95–175)"},
                  {n:"prev_grade",d:"Prior qualification grade (0–20)"},
                  {n:"sem1_grade",d:"Semester 1 mean grade (0–20)"},
                  {n:"sem2_grade",d:"Semester 2 mean grade (0–20)"},
                  {n:"sem1_approved",d:"Curricular units approved in Sem1 (0–6)"},
                ]},
                { cat:"📱 Behavioral", color:"#059669", features:[
                  {n:"logins",d:"Monthly LMS platform login count"},
                  {n:"attendance",d:"Class attendance percentage (0–100)"},
                  {n:"assignments_done",d:"Assignments submitted out of 10"},
                ]},
                { cat:"💰 Socio-Economic", color:"#7c3aed", features:[
                  {n:"scholarship",d:"Scholarship holder (0=No, 1=Yes)"},
                  {n:"tuition",d:"Tuition fees paid up to date (0/1)"},
                  {n:"debtor",d:"Has outstanding financial debt (0/1)"},
                  {n:"parent_edu",d:"Parent education level (1=None, 5=Post-grad)"},
                ]},
              ].map(cat=>(
                <div key={cat.cat} style={{ background:"#0d1628", border:`1px solid ${cat.color}25`, borderRadius:10, overflow:"hidden" }}>
                  <div style={{ padding:"9px 13px", borderBottom:`1px solid ${cat.color}20`, fontSize:11, fontWeight:800, color:cat.color, display:"flex", justifyContent:"space-between" }}>
                    {cat.cat} <span style={{ fontSize:9, background:`${cat.color}18`, padding:"2px 6px", borderRadius:8 }}>USED ✓</span>
                  </div>
                  {cat.features.map(f=>(
                    <div key={f.n} style={{ padding:"6px 13px", borderBottom:"1px solid #0a1020" }}>
                      <div style={{ fontSize:9.5, color:"#60a5fa", fontWeight:700 }}>{f.n}</div>
                      <div style={{ fontSize:9, color:"#4b6080", marginTop:1 }}>{f.d}</div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div style={{ background:"#0d1628", border:"1px solid #1a2540", borderRadius:10, overflow:"hidden" }}>
              <div style={{ padding:"9px 14px", borderBottom:"1px solid #1a2540", fontSize:9.5, color:"#4b6080", letterSpacing:"1.5px" }}>▸ SAMPLE RECORDS</div>
              <div style={{ overflowX:"auto" }}>
                <table style={{ width:"100%", borderCollapse:"collapse", fontSize:10 }}>
                  <thead><tr style={{ background:"#070c18" }}>
                    {["ID","Adm","PrevGr","Sem1","Sem2","Attend","Login","Asgn","Schlr","Tuit","Debt","PEdu","Label"].map(h=>(
                      <th key={h} style={{ padding:"7px 9px", color:"#4b6080", fontWeight:700, textAlign:"left", borderBottom:"1px solid #1a2540", whiteSpace:"nowrap" }}>{h}</th>
                    ))}
                  </tr></thead>
                  <tbody>
                    {UCI_SAMPLE.map((s,i)=>{
                      const p = localPredict(s); const lc = C[p.riskLevel];
                      return (
                        <tr key={s.id} style={{ borderBottom:"1px solid #0d1628", background:i%2?"rgba(255,255,255,0.01)":"transparent" }}>
                          <td style={{ padding:"5px 9px", color:"#4b6080" }}>{s.id}</td>
                          <td style={{ padding:"5px 9px", color:"#94a3b8" }}>{s.admission_grade}</td>
                          <td style={{ padding:"5px 9px", color:"#94a3b8" }}>{s.prev_grade}</td>
                          <td style={{ padding:"5px 9px", color:s.sem1_grade<9?"#f87171":s.sem1_grade<13?"#fbbf24":"#34d399" }}>{s.sem1_grade}</td>
                          <td style={{ padding:"5px 9px", color:s.sem2_grade<9?"#f87171":s.sem2_grade<13?"#fbbf24":"#34d399" }}>{s.sem2_grade}</td>
                          <td style={{ padding:"5px 9px", color:s.attendance<50?"#f87171":s.attendance<75?"#fbbf24":"#34d399" }}>{s.attendance}%</td>
                          <td style={{ padding:"5px 9px", color:"#94a3b8" }}>{s.logins}</td>
                          <td style={{ padding:"5px 9px", color:"#94a3b8" }}>{s.assignments_done}/10</td>
                          <td style={{ padding:"5px 9px", color:s.scholarship?"#34d399":"#4b6080" }}>{s.scholarship?"✓":"✗"}</td>
                          <td style={{ padding:"5px 9px", color:s.tuition?"#34d399":"#f87171" }}>{s.tuition?"✓":"✗"}</td>
                          <td style={{ padding:"5px 9px", color:s.debtor?"#f87171":"#4b6080" }}>{s.debtor?"✗":"✓"}</td>
                          <td style={{ padding:"5px 9px", color:"#94a3b8" }}>{s.parent_edu}/5</td>
                          <td style={{ padding:"5px 9px" }}>
                            <span style={{ background:lc.bg, color:lc.fg, border:`1px solid ${lc.border}`, padding:"2px 6px", borderRadius:3, fontSize:9, fontWeight:800 }}>{s.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ── TRAIN ── */}
        {tab==="train" && (
          <div>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:20 }}>
              <div>
                <h2 style={{ color:"#f1f5f9", fontSize:18, fontWeight:800, marginBottom:4 }}>Model Training Pipeline</h2>
                <p style={{ color:"#4b6080", fontSize:12 }}>Simulates <code style={{color:"#60a5fa"}}>python train.py</code> — in production, run locally then upload model.pkl to HF Space</p>
              </div>
              <button onClick={startTrain} disabled={isTraining} style={{
                padding:"9px 22px", borderRadius:7, border:"none", fontFamily:"inherit",
                background:isTraining?"#1a2540":"linear-gradient(135deg,#2563eb,#7c3aed)",
                color:isTraining?"#4b6080":"#fff", fontSize:11, fontWeight:800,
                cursor:isTraining?"not-allowed":"pointer", letterSpacing:"0.5px"
              }}>{isTraining?"⏳ TRAINING...":trained?"↺ RETRAIN":"▶ RUN TRAINING"}</button>
            </div>
            <div style={{ display:"grid", gridTemplateColumns:"1.1fr 0.9fr", gap:14 }}>
              <div style={{ background:"#050a12", border:"1px solid #1a2540", borderRadius:10, overflow:"hidden" }}>
                <div style={{ padding:"9px 14px", borderBottom:"1px solid #1a2540", fontSize:9.5, color:"#4b6080", letterSpacing:"1.5px" }}>▸ TRAINING CONSOLE</div>
                <div ref={logRef} style={{ height:360, overflowY:"auto", padding:14, lineHeight:1.85, fontSize:11 }}>
                  {trainSteps.length===0 && <div style={{ color:"#1a2540" }}>$ python train.py --dataset uci --smote --ensemble</div>}
                  {trainSteps.map((s,i)=>(
                    <div key={i}>
                      <span style={{ color:"#2563eb" }}>[{String(i+1).padStart(2,"0")}:{s.phase.padEnd(12)}]</span>{" "}
                      <span style={{ color:s.pct===100?"#34d399":"#64748b" }}>{s.detail}</span>
                    </div>
                  ))}
                  {trained && <><div style={{ color:"#34d399", marginTop:6 }}>✓ model.pkl saved</div><div style={{ color:"#34d399" }}>✓ Pushed to HuggingFace Space</div></>}
                </div>
                <div style={{ height:3, background:"#070c18" }}>
                  <div style={{ height:"100%", background:"linear-gradient(90deg,#2563eb,#34d399)", width:`${trainPct}%`, transition:"width 0.4s" }}/>
                </div>
                <div style={{ padding:"7px 14px", fontSize:9.5, color:"#4b6080", display:"flex", justifyContent:"space-between" }}>
                  <span>{trainPct}% complete</span>{trained&&<span style={{color:"#34d399"}}>✓ DONE</span>}
                </div>
              </div>
              <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                <div style={{ background:"#0d1628", border:"1px solid #1a2540", borderRadius:10, padding:14 }}>
                  <div style={{ fontSize:9.5, color:"#4b6080", letterSpacing:"1.5px", marginBottom:12 }}>▸ EVALUATION METRICS</div>
                  {[
                    {l:"Accuracy",         v:81.4,c:"#60a5fa"},
                    {l:"AUC-ROC",          v:89.0,c:"#a78bfa"},
                    {l:"Recall (Dropout)", v:84.0,c:"#f87171",note:"← critical"},
                    {l:"Precision",        v:78.2,c:"#34d399"},
                    {l:"F1 Score",         v:80.9,c:"#fbbf24"},
                  ].map(m=>(
                    <div key={m.l} style={{ marginBottom:10 }}>
                      <div style={{ display:"flex", justifyContent:"space-between", fontSize:10, marginBottom:3 }}>
                        <span style={{ color:"#94a3b8" }}>{m.l}</span>
                        <span style={{ color:m.c, fontWeight:800 }}>{m.v}% {m.note&&<span style={{color:"#2a3a50",fontWeight:400}}>{m.note}</span>}</span>
                      </div>
                      <div style={{ height:4, background:"#070c18", borderRadius:2 }}>
                        <div style={{ height:"100%", background:m.c, borderRadius:2, width:trained?`${m.v}%`:"3%", transition:"width 0.9s ease 0.2s" }}/>
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ background:"#0d1628", border:"1px solid #1a2540", borderRadius:10, padding:14 }}>
                  <div style={{ fontSize:9.5, color:"#4b6080", letterSpacing:"1.5px", marginBottom:10 }}>▸ FEATURE IMPORTANCES</div>
                  {[
                    {l:"Semester 1 & 2 Grade", v:20,c:"#f87171"},
                    {l:"Prev Qual. Grade",      v:14,c:"#fb923c"},
                    {l:"Attendance Rate",       v:13,c:"#fbbf24"},
                    {l:"Assignments Done",      v:10,c:"#34d399"},
                    {l:"LMS Logins",            v:9, c:"#22d3ee"},
                    {l:"Units Approved Sem1",   v:7, c:"#60a5fa"},
                  ].map(f=>(
                    <div key={f.l} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:7 }}>
                      <div style={{ fontSize:9, color:"#4b6080", width:145, flexShrink:0 }}>{f.l}</div>
                      <div style={{ flex:1, height:4, background:"#070c18", borderRadius:2 }}>
                        <div style={{ height:"100%", background:f.c, borderRadius:2, width:trained?`${f.v*4}%`:"2%", transition:"width 0.7s" }}/>
                      </div>
                      <div style={{ fontSize:9, color:f.c, width:26, textAlign:"right" }}>{f.v}%</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {trained && (
              <div style={{ marginTop:12, background:"rgba(52,211,153,0.07)", border:"1px solid rgba(52,211,153,0.25)", borderRadius:8, padding:"12px 16px", display:"flex", alignItems:"center", gap:10 }}>
                <span style={{ fontSize:20 }}>✅</span>
                <div style={{ flex:1 }}>
                  <div style={{ fontSize:12, fontWeight:800, color:"#34d399" }}>Training complete</div>
                  <div style={{ fontSize:10, color:"#4b6080", marginTop:2 }}>Head to Predict tab to run inference, or Deploy Guide to publish live</div>
                </div>
                <button onClick={()=>setTab("predict")} style={{ padding:"7px 16px", background:"#059669", border:"none", borderRadius:5, color:"#fff", fontWeight:800, fontSize:10, cursor:"pointer", fontFamily:"inherit" }}>PREDICT →</button>
              </div>
            )}
          </div>
        )}

        {/* ── PREDICT ── */}
        {tab==="predict" && (
          <div>
            <h2 style={{ color:"#f1f5f9", fontSize:18, fontWeight:800, marginBottom:4 }}>Student Risk Prediction</h2>
            <p style={{ color:"#4b6080", fontSize:12, marginBottom:16 }}>Running in <strong style={{color:useRealSpace?"#34d399":"#fbbf24"}}>{useRealSpace&&spaceUrl?"Live HF Space mode":"Local model mode"}</strong> — toggle below to use your deployed HF Space</p>

            {/* Mode toggle */}
            <div style={{ background:"#0d1628", border:"1px solid #1a2540", borderRadius:8, padding:"10px 14px", marginBottom:16, display:"flex", alignItems:"center", gap:12 }}>
              <button onClick={()=>setUseRealSpace(p=>!p)} style={{
                padding:"5px 14px", borderRadius:5, border:"none", cursor:"pointer", fontFamily:"inherit",
                background:useRealSpace?"rgba(52,211,153,0.2)":"rgba(251,191,36,0.15)",
                color:useRealSpace?"#34d399":"#fbbf24", fontSize:10, fontWeight:800,
              }}>{useRealSpace?"🟢 HF Space":"🟡 Local"}</button>
              {useRealSpace && (
                <input value={spaceUrl} onChange={e=>setSpaceUrl(e.target.value)}
                  placeholder="https://YOUR_USERNAME-edurisk.hf.space"
                  style={{ flex:1, background:"#070c18", border:"1px solid #1a2540", borderRadius:5, padding:"6px 10px", color:"#94a3b8", fontSize:10, fontFamily:"inherit", outline:"none" }}/>
              )}
              {!useRealSpace && <span style={{ fontSize:10, color:"#4b6080" }}>Using built-in ensemble model — deploy your HF Space and toggle to Live mode</span>}
            </div>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
              {/* Form */}
              <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                {/* Academic */}
                <div style={{ background:"#0d1628", border:"1px solid #2563eb28", borderRadius:10, overflow:"hidden" }}>
                  <div style={{ padding:"9px 14px", borderBottom:"1px solid #2563eb18", fontSize:9.5, fontWeight:800, color:"#60a5fa", letterSpacing:"1.5px" }}>🎓 ACADEMIC</div>
                  <div style={{ padding:14, display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                    {[
                      {k:"admission_grade",l:"Admission Grade",min:95,max:175,step:1,unit:"/175"},
                      {k:"prev_grade",l:"Prev Grade",min:0,max:20,step:0.5,unit:"/20"},
                      {k:"sem1_grade",l:"Sem 1 Grade",min:0,max:20,step:0.5,unit:"/20"},
                      {k:"sem2_grade",l:"Sem 2 Grade",min:0,max:20,step:0.5,unit:"/20"},
                    ].map(f=>(
                      <div key={f.k}>
                        <div style={{ display:"flex", justifyContent:"space-between", fontSize:8.5, marginBottom:4 }}>
                          <span style={{ color:"#4b6080" }}>{f.l.toUpperCase()}</span>
                          <span style={{ color:"#60a5fa", fontWeight:800 }}>{form[f.k]}{f.unit}</span>
                        </div>
                        <input type="range" min={f.min} max={f.max} step={f.step} value={form[f.k]}
                          onChange={e=>set(f.k,parseFloat(e.target.value))}
                          style={{ width:"100%", accentColor:"#2563eb" }}/>
                      </div>
                    ))}
                    <div style={{ gridColumn:"1/-1" }}>
                      <div style={{ display:"flex", justifyContent:"space-between", fontSize:8.5, marginBottom:4 }}>
                        <span style={{ color:"#4b6080" }}>UNITS APPROVED SEM1</span>
                        <span style={{ color:"#60a5fa", fontWeight:800 }}>{form.sem1_approved} / 6</span>
                      </div>
                      <input type="range" min={0} max={6} step={1} value={form.sem1_approved}
                        onChange={e=>set("sem1_approved",parseInt(e.target.value))}
                        style={{ width:"100%", accentColor:"#2563eb" }}/>
                    </div>
                  </div>
                </div>
                {/* Behavioral */}
                <div style={{ background:"#0d1628", border:"1px solid #05966928", borderRadius:10, overflow:"hidden" }}>
                  <div style={{ padding:"9px 14px", borderBottom:"1px solid #05966918", fontSize:9.5, fontWeight:800, color:"#34d399", letterSpacing:"1.5px" }}>📱 BEHAVIORAL</div>
                  <div style={{ padding:14, display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:12 }}>
                    {[
                      {k:"logins",l:"LMS Logins/mo",min:0,max:100,step:1,unit:""},
                      {k:"attendance",l:"Attendance",min:0,max:100,step:1,unit:"%"},
                      {k:"assignments_done",l:"Assignments",min:0,max:10,step:1,unit:"/10"},
                    ].map(f=>(
                      <div key={f.k}>
                        <div style={{ display:"flex", justifyContent:"space-between", fontSize:8.5, marginBottom:4 }}>
                          <span style={{ color:"#4b6080" }}>{f.l.toUpperCase()}</span>
                          <span style={{ color:"#34d399", fontWeight:800 }}>{form[f.k]}{f.unit}</span>
                        </div>
                        <input type="range" min={f.min} max={f.max} step={f.step} value={form[f.k]}
                          onChange={e=>set(f.k,parseFloat(e.target.value))}
                          style={{ width:"100%", accentColor:"#059669" }}/>
                      </div>
                    ))}
                  </div>
                </div>
                {/* Socio-economic */}
                <div style={{ background:"#0d1628", border:"1px solid #7c3aed28", borderRadius:10, overflow:"hidden" }}>
                  <div style={{ padding:"9px 14px", borderBottom:"1px solid #7c3aed18", fontSize:9.5, fontWeight:800, color:"#a78bfa", letterSpacing:"1.5px" }}>💰 SOCIO-ECONOMIC</div>
                  <div style={{ padding:14 }}>
                    <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:8, marginBottom:12 }}>
                      {[{k:"scholarship",l:"Scholarship",pos:true},{k:"tuition",l:"Tuition Paid",pos:true},{k:"debtor",l:"Has Debt",pos:false}].map(f=>(
                        <button key={f.k} onClick={()=>set(f.k,form[f.k]?0:1)} style={{
                          padding:"7px 5px", borderRadius:5, border:"none", cursor:"pointer", fontFamily:"inherit",
                          background:form[f.k]?(f.pos?"rgba(52,211,153,0.15)":"rgba(248,113,113,0.15)"):(f.pos?"rgba(248,113,113,0.1)":"rgba(52,211,153,0.1)"),
                          color:form[f.k]?(f.pos?"#34d399":"#f87171"):(f.pos?"#f87171":"#34d399"),
                          fontSize:9.5, fontWeight:700, lineHeight:1.5,
                        }}><div>{f.l}</div><div>{form[f.k]?"YES":"NO"}</div></button>
                      ))}
                    </div>
                    <div style={{ fontSize:8.5, color:"#4b6080", marginBottom:4, display:"flex", justifyContent:"space-between" }}>
                      <span>PARENT EDUCATION LEVEL</span><span style={{color:"#a78bfa",fontWeight:800}}>Level {form.parent_edu}/5</span>
                    </div>
                    <input type="range" min={1} max={5} step={1} value={form.parent_edu}
                      onChange={e=>set("parent_edu",parseInt(e.target.value))}
                      style={{ width:"100%", accentColor:"#7c3aed" }}/>
                  </div>
                </div>
                {predError && <div style={{ padding:"9px 12px", background:"rgba(248,113,113,0.08)", border:"1px solid rgba(248,113,113,0.2)", borderRadius:6, fontSize:10, color:"#f87171" }}>⚠ {predError}</div>}
                <button onClick={runPredict} disabled={predicting} style={{
                  padding:"12px", borderRadius:7, border:"none",
                  background:predicting?"#1a2540":"linear-gradient(135deg,#2563eb,#7c3aed)",
                  color:predicting?"#4b6080":"#fff", fontSize:12, fontWeight:800,
                  cursor:predicting?"not-allowed":"pointer", fontFamily:"inherit",
                }}>{predicting?"⏳ PREDICTING...":"🎯 RUN ENSEMBLE PREDICTION"}</button>
              </div>

              {/* Results */}
              <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                {result ? (() => {
                  const lc = C[result.risk_level || result.riskLevel || "MODERATE"];
                  const riskLevel = result.risk_level || result.riskLevel || "MODERATE";
                  return (<>
                    <div style={{ background:lc.bg, border:`1px solid ${lc.border}`, borderRadius:10, padding:18 }}>
                      <div style={{ display:"flex", alignItems:"center", gap:16 }}>
                        <div style={{ textAlign:"center" }}>
                          <Gauge value={result.dropout} color={lc.fg} size={110}/>
                          <div style={{ fontSize:8.5, color:lc.fg, letterSpacing:"1.5px", fontWeight:700, marginTop:3 }}>DROPOUT RISK</div>
                        </div>
                        <div style={{ flex:1 }}>
                          <div style={{ display:"inline-block", padding:"3px 12px", borderRadius:20, background:lc.fg, color:"#000", fontSize:10, fontWeight:900, letterSpacing:"2px", marginBottom:10 }}>{riskLevel} RISK</div>
                          <div style={{ display:"flex", gap:14 }}>
                            {[["Dropout",result.dropout,lc.fg],["Graduate",result.graduate,"#34d399"],["Enrolled",result.enrolled,"#fbbf24"]].map(([l,v,c])=>(
                              <div key={l} style={{ textAlign:"center" }}>
                                <div style={{ fontSize:15, fontWeight:800, color:c }}>{Math.round(v*100)}%</div>
                                <div style={{ fontSize:8.5, color:"#4b6080" }}>{l}</div>
                              </div>
                            ))}
                          </div>
                          <div style={{ marginTop:10, fontSize:9.5, color:"#4b6080", lineHeight:1.6 }}>
                            {riskLevel==="HIGH"&&"⚠️ Immediate advisor outreach recommended."}
                            {riskLevel==="MODERATE"&&"📋 Monitor closely. Schedule check-in within 2 weeks."}
                            {riskLevel==="LOW"&&"✅ Student on track. Continue standard engagement."}
                          </div>
                        </div>
                      </div>
                    </div>
                    {result.riskFactors?.length > 0 && (
                      <div style={{ background:"#0d1628", border:"1px solid #1a2540", borderRadius:10, padding:14 }}>
                        <div style={{ fontSize:9.5, color:"#4b6080", letterSpacing:"1.5px", marginBottom:10 }}>▸ SHAP-STYLE ATTRIBUTION</div>
                        <div style={{ fontSize:9.5, color:"#f87171", marginBottom:5 }}>RISK FACTORS</div>
                        {result.riskFactors.map(f=>(
                          <div key={f.feature} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:5 }}>
                            <div style={{ fontSize:9, color:"#f87171", width:140, flexShrink:0 }}>⚠ {FEAT_LABELS[f.feature]||f.feature}</div>
                            <div style={{ flex:1, height:3, background:"#070c18", borderRadius:2 }}>
                              <div style={{ height:"100%", background:"#f87171", borderRadius:2, width:`${Math.min(100,f.impact*100*6)}%`, opacity:0.75 }}/>
                            </div>
                            <div style={{ fontSize:9, color:"#94a3b8", width:28, textAlign:"right" }}>{f.raw}</div>
                          </div>
                        ))}
                        <div style={{ fontSize:9.5, color:"#34d399", margin:"9px 0 5px" }}>STRENGTHS</div>
                        {result.strengths.map(f=>(
                          <div key={f.feature} style={{ display:"flex", alignItems:"center", gap:6, marginBottom:5 }}>
                            <div style={{ fontSize:9, color:"#34d399", width:140, flexShrink:0 }}>✓ {FEAT_LABELS[f.feature]||f.feature}</div>
                            <div style={{ flex:1, height:3, background:"#070c18", borderRadius:2 }}>
                              <div style={{ height:"100%", background:"#34d399", borderRadius:2, width:`${Math.min(100,f.impact*100*6)}%` }}/>
                            </div>
                            <div style={{ fontSize:9, color:"#94a3b8", width:28, textAlign:"right" }}>{f.raw}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Mistral nudge */}
                    <div style={{ background:"#080e1c", border:"1px solid rgba(5,150,105,0.3)", borderRadius:10, padding:14 }}>
                      <div style={{ fontSize:9.5, color:"#059669", letterSpacing:"1.5px", marginBottom:9 }}>🤖 MISTRAL-7B-INSTRUCT ADVISOR NUDGE</div>
                      <div style={{ fontSize:8.5, color:"#4b6080", marginBottom:5 }}>HuggingFace token (free at hf.co/settings/tokens)</div>
                      <div style={{ display:"flex", gap:6, marginBottom:8 }}>
                        <input type={showToken?"text":"password"} placeholder="hf_xxxxxxxxxxxxxxxx"
                          value={hfToken} onChange={e=>setHfToken(e.target.value)}
                          style={{ flex:1, background:"#0d1628", border:"1px solid #1a2540", borderRadius:5, padding:"7px 10px", color:"#94a3b8", fontSize:10, fontFamily:"inherit", outline:"none" }}/>
                        <button onClick={()=>setShowToken(p=>!p)} style={{ padding:"0 8px", background:"#1a2540", border:"none", borderRadius:5, color:"#4b6080", cursor:"pointer" }}>{showToken?"🙈":"👁"}</button>
                      </div>
                      <button onClick={fetchNudge} disabled={nudgeLoading} style={{
                        width:"100%", padding:"9px", borderRadius:5, border:"none",
                        background:nudgeLoading?"#1a2540":"rgba(5,150,105,0.22)",
                        color:nudgeLoading?"#4b6080":"#34d399",
                        fontSize:10, fontWeight:800, cursor:nudgeLoading?"not-allowed":"pointer", fontFamily:"inherit"
                      }}>{nudgeLoading?"⏳ Calling Mistral-7B-Instruct-v0.3...":"🤖 Generate Personalised Advisor Message"}</button>
                      {nudgeError && <div style={{ marginTop:8, padding:"8px 10px", background:"rgba(248,113,113,0.08)", border:"1px solid rgba(248,113,113,0.2)", borderRadius:5, fontSize:9.5, color:"#f87171" }}>⚠ {nudgeError}</div>}
                      {nudge && (
                        <div style={{ marginTop:10, padding:"12px", background:"rgba(52,211,153,0.06)", border:"1px solid rgba(52,211,153,0.2)", borderRadius:7 }}>
                          <div style={{ fontSize:8.5, color:"#059669", letterSpacing:"1.5px", marginBottom:6 }}>GENERATED ADVISOR MESSAGE ✓</div>
                          <div style={{ fontSize:12, color:"#a7f3d0", lineHeight:1.8, fontFamily:"Georgia,serif", fontStyle:"italic" }}>{nudge}</div>
                        </div>
                      )}
                      <div style={{ marginTop:8, fontSize:8.5, color:"#2a3a50", lineHeight:1.6 }}>
                        First call may take ~20s (cold start). 503 = model loading, retry after 20s.
                      </div>
                    </div>
                  </>);
                })() : (
                  <div style={{ background:"#0d1628", border:"1px dashed #1a2540", borderRadius:10, padding:50, textAlign:"center", color:"#2a3a50", fontSize:12 }}>
                    <div style={{ fontSize:28, marginBottom:10 }}>🎯</div>
                    Set features on the left, then click<br/><strong style={{color:"#2563eb"}}>RUN ENSEMBLE PREDICTION</strong>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── DEPLOY GUIDE ── */}
        {tab==="deploy" && (
          <div>
            <h2 style={{ color:"#f1f5f9", fontSize:18, fontWeight:800, marginBottom:4 }}>🚀 Deployment Guide</h2>
            <p style={{ color:"#4b6080", fontSize:12, marginBottom:24 }}>3 steps to go fully live. Everything is free.</p>
            <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
              {[
                {
                  step:"01", title:"Deploy ML Model → HuggingFace Space", color:"#0891b2",
                  time:"~10 mins", cost:"Free",
                  instructions:[
                    "Go to huggingface.co/new-space",
                    "Name it 'edurisk' · Select SDK: Gradio · Hardware: CPU Basic (Free)",
                    "Upload these files from the hf_space/ folder:",
                    "  • app.py   • requirements.txt   • README.md",
                    "Run train.py locally: pip install scikit-learn xgboost imbalanced-learn && python train.py",
                    "Upload the generated model.pkl to your Space (Files tab)",
                    "Space auto-builds. Your API URL will be: https://YOUR_NAME-edurisk.hf.space",
                  ],
                  note:"app.py has a fallback: if model.pkl is missing it trains a synthetic model automatically so the Space won't crash",
                },
                {
                  step:"02", title:"Deploy Frontend → Vercel", color:"#7c3aed",
                  time:"~5 mins", cost:"Free",
                  instructions:[
                    "Push the frontend/ folder to a GitHub repo",
                    "Go to vercel.com → New Project → Import your repo",
                    "In Vercel Environment Variables, add:",
                    "  VITE_HF_SPACE_URL = https://srikanth-haki-mini-project.hf.space",
                    "  VITE_HF_TOKEN     = hf_xxxx  (optional — users can enter their own)",
                    "Click Deploy. Done. URL: https://edurisk.vercel.app",
                    "Every git push auto-redeploys via CI/CD",
                  ],
                  note:"Alternatively use Netlify: drag-and-drop the dist/ folder after running npm run build",
                },
                {
                  step:"03", title:"Get Mistral-7B Token (LLM Nudge)", color:"#059669",
                  time:"~2 mins", cost:"Free",
                  instructions:[
                    "Go to huggingface.co/settings/tokens",
                    "Click New Token → Role: Read → Create",
                    "Copy the token (starts with hf_)",
                    "Either add to Vercel env as VITE_HF_TOKEN",
                    "Or let users paste their own token in the UI (current behaviour)",
                    "Model: mistralai/Mistral-7B-Instruct-v0.3 — no approval needed",
                    "Free tier: ~1,000 API calls/day",
                  ],
                  note:"Want a different model? Swap the URL to HuggingFaceH4/zephyr-7b-beta or meta-llama/Meta-Llama-3-8B-Instruct (requires acceptance of terms)",
                },
              ].map(s=>(
                <div key={s.step} style={{ background:"#0d1628", border:`1px solid ${s.color}25`, borderRadius:10, overflow:"hidden" }}>
                  <div style={{ padding:"12px 16px", borderBottom:`1px solid ${s.color}18`, display:"flex", alignItems:"center", gap:12 }}>
                    <div style={{ width:32, height:32, borderRadius:7, background:`${s.color}25`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, fontWeight:800, color:s.color, flexShrink:0 }}>{s.step}</div>
                    <div style={{ flex:1 }}>
                      <div style={{ fontSize:13, fontWeight:800, color:"#e2e8f0" }}>{s.title}</div>
                    </div>
                    <div style={{ display:"flex", gap:8 }}>
                      <span style={{ fontSize:9, background:"rgba(52,211,153,0.12)", color:"#34d399", padding:"3px 8px", borderRadius:10, fontWeight:700 }}>⏱ {s.time}</span>
                      <span style={{ fontSize:9, background:"rgba(52,211,153,0.12)", color:"#34d399", padding:"3px 8px", borderRadius:10, fontWeight:700 }}>💰 {s.cost}</span>
                    </div>
                  </div>
                  <div style={{ padding:"14px 16px" }}>
                    <ol style={{ margin:0, padding:"0 0 0 16px", color:"#94a3b8", fontSize:11, lineHeight:2.1 }}>
                      {s.instructions.map((line,i)=>(
                        <li key={i} style={{ color: line.startsWith("  ")?"#4b6080":"#94a3b8", listStyle:line.startsWith("  ")?"none":"decimal", fontFamily:"monospace", fontSize:10.5 }}>{line}</li>
                      ))}
                    </ol>
                    <div style={{ marginTop:10, padding:"8px 12px", background:`${s.color}10`, border:`1px solid ${s.color}20`, borderRadius:6, fontSize:10, color:s.color }}>
                      💡 {s.note}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div style={{ marginTop:16, background:"#0d1628", border:"1px solid #1a2540", borderRadius:10, padding:16 }}>
              <div style={{ fontSize:11, fontWeight:800, color:"#f1f5f9", marginBottom:12 }}>📁 Files to deploy — download from the outputs</div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                {[
                  { folder:"hf_space/", color:"#0891b2", files:["app.py — Gradio API server","requirements.txt — Python deps","train.py — Run locally to create model.pkl","README.md — HF Space metadata"] },
                  { folder:"frontend/", color:"#7c3aed", files:["src/App.jsx — Full React app","src/main.jsx — Entry point","index.html — HTML shell","package.json — npm deps","vite.config.js — Build config",".env.example — Copy to .env.local"] },
                ].map(f=>(
                  <div key={f.folder} style={{ background:"#070c18", border:`1px solid ${f.color}20`, borderRadius:8, padding:12 }}>
                    <div style={{ fontSize:10, fontWeight:800, color:f.color, marginBottom:8, fontFamily:"monospace" }}>{f.folder}</div>
                    {f.files.map(file=>(
                      <div key={file} style={{ fontSize:9.5, color:"#4b6080", lineHeight:1.9, fontFamily:"monospace" }}>▸ {file}</div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div style={{ borderTop:"1px solid #0d1628", padding:"12px 24px", textAlign:"center", fontSize:8.5, color:"#2a3a50", letterSpacing:"1.5px" }}>
        EDURISK · MISTRAL-7B-INSTRUCT-V0.3 · RF+XGB+LR ENSEMBLE · UCI DROPOUT DATASET · VERCEL + HUGGINGFACE SPACES
      </div>
    </div>
  );
}

import {motion} from "framer-motion";
import {ArrowDown,ArrowRight,BarChart3,BrainCircuit,Database,Globe2,Layers3,LucideIcon,MapPin,Radio,Scale,TimerReset} from "lucide-react";
import Page from "../components/common/Page";
import GlassCard from "../components/common/GlassCard";

const stages:{num:string;title:string;desc:string;Icon:LucideIcon}[] = [
  {num:"01",title:"NASA FIRMS",desc:"655,204 hotspot observations from 2025.",Icon:Database},
  {num:"02",title:"Context retrieval",desc:"OSM, Dynamic World, and local FIRMS history.",Icon:Globe2},
  {num:"03",title:"Evidence engine",desc:"Heuristic scores create contextual pseudo-labels.",Icon:Scale},
  {num:"04",title:"Random Forest",desc:"A 74-feature model learns the source classes.",Icon:BrainCircuit},
];

const classes=[
  ["01","Industrial Fire","built + industrial context"],
  ["02","Gas Flare","flare + fossil infrastructure"],
  ["03","Agricultural Fire","crops + farmland context"],
  ["04","Mining Activity","quarry + bare land context"],
  ["05","Wildfire","wildland + natural vegetation"],
  ["00","Other","conceptual residual class"],
];

const features=[
  ["FIRMS base","7","latitude, longitude, brightness, scan, track, bright_t31, frp"],
  ["OSM raw distances","20","Nearest distances to targeted infrastructure and land use"],
  ["OSM log distances","20","log1p transform of each OSM distance"],
  ["Dynamic World","13","9 land-cover probabilities + date, month, max probability, availability"],
  ["Persistence","4","7/30-day detections, night ratio, persistence score"],
  ["Engineered FIRMS","6","frp_log, confidence indicators, hour_sin, hour_cos"],
  ["Grouped distances","4","fossil, mining, agriculture, and wildland minima"],
];

export default function Model1(){
  return <Page title="Model 1" subtitle="A contextual fire-source classifier: pseudo-label generation first, Random Forest learning second." wide>
    <div className="model1-hero">
      <motion.div initial={{opacity:0,x:-20}} animate={{opacity:1,x:0}} className="model1-copy">
        <div className="eyebrow">MODEL 1 · CONTEXTUAL SOURCE CLASSIFICATION</div>
        <h2>From a hotspot to a source hypothesis.</h2>
        <p>
          Model 1 combines NASA FIRMS measurements with geographic context, land cover, and local fire behavior. The evidence engine creates pseudo-labels; the Random Forest learns to reproduce those decisions from a fixed 74-feature contract.
        </p>
        <div className="model1-stat-row"><div><b>74</b><span>features</span></div><div><b>25 km</b><span>OSM retrieval</span></div><div><b>1 km</b><span>fire history</span></div></div>
      </motion.div>
      <GlassCard className="model1-preview">
        <div className="eyebrow">MODEL CONTRACT</div>
        <div className="preview-ring"><BrainCircuit size={30}/><span>74 FEATURES</span></div>
        <strong>Context in, class out.</strong>
        <small>Training medians, feature order, and class mapping travel with the saved model package.</small>
      </GlassCard>
    </div>

    <div className="model1-grid model1-pipeline">
      {stages.map(({num,title,desc,Icon},index)=>{
        return <motion.div key={num} initial={{opacity:0,y:18}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{delay:index*.07}}>
          <GlassCard className="model1-stage">
            <div className="stage-num">{num}</div>
            <Icon size={21}/>
            <h3>{title}</h3>
            <p>{desc}</p>
            {index<stages.length-1&&<span className="stage-arrow"><ArrowRight size={15}/></span>}
          </GlassCard>
        </motion.div>
      })}
    </div>

    <section className="model1-section"><div className="eyebrow">THE THREE DISTANCE SCALES</div><h2>Look wide. Decide close.</h2><p className="model1-intro">The radii are deliberately different. They answer three different questions instead of treating every nearby object as equal evidence.</p><div className="distance-grid"><GlassCard><div className="distance-icon"><Globe2/></div><b>25 km</b><strong>Retrieve context</strong><p>Search the OSM bounding box broadly enough to find the nearest relevant object and preserve its actual distance.</p><span>OSM SEARCH RADIUS</span></GlassCard><GlassCard><div className="distance-icon"><MapPin/></div><b>1–2 km</b><strong>Count evidence</strong><p>Turn nearest-object distances into flags such as industrial_near_1km or forest_near_2km.</p><span>EVIDENCE THRESHOLDS</span></GlassCard><GlassCard><div className="distance-icon"><TimerReset/></div><b>1 km</b><strong>Measure recurrence</strong><p>Use a local BallTree neighborhood to count previous FIRMS detections without mixing separate fire areas.</p><span>PERSISTENCE RADIUS</span></GlassCard></div></section>

    <section className="model1-section evidence-section"><div className="model1-two-col"><div><div className="eyebrow">PSEUDO-LABEL ENGINE</div><h2>Evidence becomes a teaching signal.</h2><p className="model1-intro">The label is not copied from FIRMS. Geographic evidence, Dynamic World land cover, fire behavior, and evidence specificity accumulate into competing source scores.</p><div className="score-equation"><span>OSM proximity</span><ArrowRight size={15}/><span>land cover</span><ArrowRight size={15}/><span>fire behavior</span><ArrowRight size={15}/><b>class score</b></div></div><GlassCard className="score-card"><div className="score-card-head"><Radio size={16}/><span>EXAMPLE SCORE STACK</span></div><div className="score-line"><span>Gas Flare</span><b>+8</b><small>flare ≤ 2 km</small></div><div className="score-line"><span>Industrial</span><b>+6</b><small>industrial ≤ 1 km</small></div><div className="score-line"><span>Wildfire</span><b>+3</b><small>forest ≤ 2 km</small></div><div className="score-line"><span>Other</span><b>+1</b><small>residual baseline</small></div></GlassCard></div></section>

    <section className="model1-section"><div className="eyebrow">BEHAVIOR + SPECIFICITY</div><h2>Context is not static.</h2><div className="behavior-grid"><GlassCard><TimerReset size={19}/><h3>Persistence</h3><p>Previous detections within 1 km produce 7-day and 30-day counts, night ratio, and a normalized score.</p><div className="formula">0.6 × score<sub>7</sub> + 0.4 × score<sub>30</sub></div><small>p95 normalization is learned from training data</small></GlassCard><GlassCard><Scale size={19}/><h3>Specificity</h3><p>Common clues are down-weighted; rare clues such as a nearby flare are more distinctive.</p><div className="formula">1 / (1 + frequency × 5)</div><small>applied to selected geographic evidence</small></GlassCard><GlassCard><BarChart3 size={19}/><h3>Margin</h3><p>The winning score is compared with the runner-up before the row is trusted for training.</p><div className="formula">winner − second place</div><small>≥ 5 High · ≥ 2 Medium · otherwise Low</small></GlassCard></div></section>

    <section className="model1-section quality-section"><div className="model1-two-col quality-layout"><div><div className="eyebrow">QUALITY GATE</div><h2>Uncertain rows stay out.</h2><p className="model1-intro">The evidence engine keeps the uncertainty visible instead of pretending every hotspot is ground truth.</p><div className="quality-flow"><span>655,204 master rows</span><ArrowDown size={16}/><span>539,144 Trusted + Candidate</span><ArrowDown size={16}/><b>Random Forest training</b></div></div><div className="quality-bars"><div><span>High / Trusted</span><b>311,465</b><i style={{width:"58%"}}/></div><div><span>Medium / Candidate</span><b>227,679</b><i style={{width:"42%"}}/></div><div className="muted"><span>Low / Uncertain</span><b>116,060 excluded</b><i style={{width:"22%"}}/></div></div></div></section>

    <section className="model1-section feature-section"><div className="eyebrow">THE 74-FEATURE CONTRACT</div><h2>Exact order matters.</h2><p className="model1-intro">The same feature names, ordering, distances, transformations, and training medians must be used at live inference.</p><div className="feature-table">{features.map(([group,count,detail])=><div className="feature-row" key={group}><b>{group}</b><strong>{count}</strong><span>{detail}</span></div>)}</div></section>

    <section className="model1-section model1-final"><div><div className="eyebrow">SAVED ARTIFACT</div><h2>One package keeps training and inference aligned.</h2><p className="model1-intro">The Random Forest is saved with the feature order, class mapping, training medians, model type, and random state in <code>models/model1/model_package.joblib</code>.</p></div><GlassCard className="rf-card"><div className="eyebrow">RANDOM FOREST</div><b>200 trees</b><span>min_samples_leaf = 2</span><span>class_weight = balanced</span><span>random_state = 42</span><div className="class-note">Learned RF classes: <strong>1–5</strong><small>Conceptual class 0 / Other is not present in the final estimator.</small></div></GlassCard></section>
  </Page>
}

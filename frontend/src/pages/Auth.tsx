import {FormEvent, useState} from "react";
import {Link, useNavigate, useSearchParams} from "react-router-dom";
import {ArrowRight, Building2, LockKeyhole, Mail, Satellite, UserRound} from "lucide-react";

type AuthMode = "login" | "signup";
type AccountType = "user" | "department";

export default function Auth(){
  const navigate=useNavigate();
  const [params]=useSearchParams();
  const [mode,setMode]=useState<AuthMode>(params.get("mode")==="signup"?"signup":"login");
  const [accountType,setAccountType]=useState<AccountType>("user");
  const [submitted,setSubmitted]=useState(false);

  function handleSubmit(event:FormEvent<HTMLFormElement>){
    event.preventDefault();
    localStorage.setItem("firms_auth_session",JSON.stringify({accountType,mode,createdAt:new Date().toISOString()}));
    setSubmitted(true);
    window.setTimeout(()=>navigate("/dashboard"),500);
  }

  function switchMode(nextMode:AuthMode){
    setMode(nextMode);
    setSubmitted(false);
  }

  return <main className="auth-page">
    <section className="auth-visual">
      <Link className="brand auth-brand" to="/">
        <span className="brandmark"><Satellite size={17}/></span>
        <span>FIRMS <b>Fire Intelligence</b></span>
      </Link>
      <div className="auth-visual-copy">
        <div className="eyebrow"><span/> SOURCE INTELLIGENCE PLATFORM</div>
        <h1>See the signal<br/><em>before it spreads.</em></h1>
        <p>Bring satellite observations, field context, and model confidence into one shared intelligence workspace.</p>
      </div>
      <div className="auth-signal"><i/> LIVE MODEL STATUS <strong>98.4%</strong><span>classification confidence</span></div>
    </section>
    <section className="auth-panel">
      <div className="auth-panel-inner">
        <div className="auth-mobile-brand"><Satellite size={16}/> FIRMS / ACCESS</div>
        <div className="auth-heading">
          <div className="auth-tabs" role="tablist" aria-label="Authentication">
            <button className={mode==="login"?"active":""} onClick={()=>switchMode("login")} type="button">Sign in</button>
            <button className={mode==="signup"?"active":""} onClick={()=>switchMode("signup")} type="button">Create account</button>
          </div>
          <h2>{mode==="login"?"Welcome back.":"Start your workspace."}</h2>
          <p>{mode==="login"?"Access your fire intelligence workspace.":"Choose the account that fits how your team works."}</p>
        </div>
        {mode==="signup"&&<div className="account-picker" role="group" aria-label="Account type">
          <button className={accountType==="user"?"selected":""} onClick={()=>setAccountType("user")} type="button"><UserRound size={17}/><span><b>Individual</b><small>Explore and analyze events</small></span></button>
          <button className={accountType==="department"?"selected":""} onClick={()=>setAccountType("department")} type="button"><Building2 size={17}/><span><b>Department</b><small>Coordinate a response team</small></span></button>
        </div>}
        <form className="auth-form" onSubmit={handleSubmit}>
          {mode==="signup"&&<label><span>{accountType==="department"?"Department name":"Full name"}</span><div className="input-wrap"><Building2 size={16}/><input required placeholder={accountType==="department"?"e.g. State Emergency Office":"e.g. Alex Morgan"}/></div></label>}
          <label><span>Work email</span><div className="input-wrap"><Mail size={16}/><input type="email" required placeholder="you@organization.org"/></div></label>
          {mode==="signup"&&accountType==="department"&&<label><span>Department code <small>Optional</small></span><div className="input-wrap"><Building2 size={16}/><input placeholder="e.g. CAL-FS"/></div></label>}
          <label><span>Password</span><div className="input-wrap"><LockKeyhole size={16}/><input type="password" required minLength={8} placeholder="At least 8 characters"/></div></label>
          {mode==="login"&&<div className="form-meta"><label className="check"><input type="checkbox"/> <span>Remember me</span></label><button type="button" className="text-button">Forgot password?</button></div>}
          {mode==="signup"&&<label className="check terms"><input type="checkbox" required/> <span>I agree to the terms and data use policy.</span></label>}
          <button className="btn primary auth-submit" type="submit">{submitted?"Opening workspace...":mode==="login"?"Sign in to workspace":"Create workspace"}<ArrowRight size={17}/></button>
        </form>
        <p className="auth-footer">{mode==="login"?"New to FIRMS? ":"Already have access? "}<button type="button" className="text-button" onClick={()=>switchMode(mode==="login"?"signup":"login")}>{mode==="login"?"Create an account":"Sign in"}</button></p>
        <small className="auth-demo-note">Demo access is local for now. Connect your API to enable production authentication.</small>
      </div>
    </section>
  </main>
}

import {Navigate,Routes,Route} from "react-router-dom";
import type {ReactNode} from "react";
import MainLayout from "../layouts/MainLayout";
import Landing from "../pages/Landing";
import Dashboard from "../pages/Dashboard";
import Events from "../pages/Events";
import EventDetail from "../pages/EventDetail";
import MapPage from "../pages/MapPage";
import Analytics from "../pages/Analytics";
import Model from "../pages/Model";
import Model1 from "../pages/Model1";
import ModelPipeline from "../pages/ModelPipeline";
import Analyze from "../pages/Analyze";
import About from "../pages/About";
import Settings from "../pages/Settings";
import Auth from "../pages/Auth";
import AIPrediction from "../pages/AIPrediction";

function RequireAuth({children}:{children:ReactNode}){
  return localStorage.getItem("firms_auth_session")?<>{children}</>:<Navigate to="/login" replace/>;
}

export default function AppRoutes(){
  return <Routes><Route path="/login" element={<Auth/>}/><Route path="/signup" element={<Auth/>}/><Route element={<MainLayout/>}>
    <Route path="/" element={<Landing/>}/>
    <Route path="/dashboard" element={<RequireAuth><Dashboard/></RequireAuth>}/>
    <Route path="/events" element={<RequireAuth><Events/></RequireAuth>}/>
    <Route path="/events/:eventId" element={<RequireAuth><EventDetail/></RequireAuth>}/>
    <Route path="/map" element={<RequireAuth><MapPage/></RequireAuth>}/>
    <Route path="/analytics" element={<RequireAuth><Analytics/></RequireAuth>}/>
    <Route path="/model" element={<RequireAuth><Model/></RequireAuth>}/>
    <Route path="/model/1" element={<RequireAuth><Model1/></RequireAuth>}/>
    <Route path="/model/pipeline" element={<RequireAuth><ModelPipeline/></RequireAuth>}/>
    <Route path="/analyze" element={<RequireAuth><Analyze/></RequireAuth>}/>
    <Route path="/about" element={<RequireAuth><About/></RequireAuth>}/>
    <Route path="/settings" element={<RequireAuth><Settings/></RequireAuth>}/>
    <Route
  path="/ai-prediction"
  element={
    <RequireAuth>
      <AIPrediction />
    </RequireAuth>
  }
/>
  </Route></Routes>
}

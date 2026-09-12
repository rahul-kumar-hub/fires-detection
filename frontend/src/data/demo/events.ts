import type {FireEvent,LabelingFunction} from "../../types/fire";
const lf=(name:string,category:any,active:boolean,agrees:boolean,description:string):LabelingFunction=>({name,category,active,agrees,description});
const evidence:LabelingFunction[]=[
 lf("LF_WF_STRONG_FOREST","WILDFIRE",true,true,"Strong forest evidence"),
 lf("LF_WF_REPEATED_NATURAL","WILDFIRE",true,true,"Repeated natural activity"),
 lf("LF_GAS_1KM","GAS",false,false,"No gas flare signal"),
 lf("LF_IND_SCORE","INDUSTRIAL",false,false,"No industrial signal")
];
const coords=[[23.6421,85.3012],[23.676,85.283],[23.61,85.34],[23.72,85.22],[23.57,85.27],[23.69,85.37],[23.53,85.19],[23.75,85.31]];
const classes=["WILDFIRE","AGRICULTURE","INDUSTRIAL","MINING","GAS"] as const;
export const demoEvents:FireEvent[]=Array.from({length:28},(_,i)=>{
 const c=classes[i%classes.length], p= i%5===0?.947: i%3===0?.83:.64+(i%7)/40;
 return {eventId:`FIRMS-${String(123+i).padStart(6,"0")}`,date:`2026-09-${String(1+(i%9)).padStart(2,"0")} ${String(10+(i%10)).padStart(2,"0")}:24`,latitude:coords[i%coords.length][0]+(i%4)*.004,longitude:coords[i%coords.length][1]+(i%3)*.005,location:["Patratu, Jharkhand","Ramgarh, Jharkhand","Bokaro, Jharkhand","Ranchi, Jharkhand"][i%4],prediction:{predictedClass:c,confidence:p,probabilities:{INDUSTRIAL:c==="INDUSTRIAL"?.78:.05,GAS:c==="GAS"?.82:.04,AGRICULTURE:c==="AGRICULTURE"?.74:.06,MINING:c==="MINING"?.79:.04,WILDFIRE:c==="WILDFIRE"?.947:.05}},lfAgreement: i%5===0?4:3,activeLFs:i%5===0?5:4,agreeingLFs:i%5===0?4:3,disagreeingLFs:i%5===0?1:1,evidence};
});
export const getDemoEvent=(id:string)=>demoEvents.find(e=>e.eventId===id)??demoEvents[0];

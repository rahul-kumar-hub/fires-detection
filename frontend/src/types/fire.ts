export type FireClass = "INDUSTRIAL"|"GAS"|"AGRICULTURE"|"MINING"|"WILDFIRE";
export interface Prediction { predictedClass: FireClass; confidence: number; probabilities: Record<FireClass, number>; }
export interface LabelingFunction { name:string; category:FireClass; active:boolean; agrees:boolean; description:string; }
export interface FireEvent { eventId:string; date:string; latitude:number; longitude:number; location:string; prediction:Prediction; lfAgreement:number; activeLFs:number; agreeingLFs:number; disagreeingLFs:number; evidence:LabelingFunction[]; }

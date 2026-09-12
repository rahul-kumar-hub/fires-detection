import type {FireClass} from "../types/fire";
export const FIRE_CLASSES: Record<FireClass,{label:string; short:string; icon:string}> = {
 INDUSTRIAL:{label:"Industrial",short:"INDUSTRIAL",icon:"Factory"}, GAS:{label:"Gas",short:"GAS",icon:"Flame"},
 AGRICULTURE:{label:"Agriculture",short:"AGRICULTURE",icon:"Wheat"}, MINING:{label:"Mining",short:"MINING",icon:"Pickaxe"},
 WILDFIRE:{label:"Wildfire",short:"WILDFIRE",icon:"Trees"}
};
export const CLASS_ORDER: FireClass[]=["INDUSTRIAL","GAS","AGRICULTURE","MINING","WILDFIRE"];
export const CONFIDENCE_BANDS={high:0.85,medium:0.6};
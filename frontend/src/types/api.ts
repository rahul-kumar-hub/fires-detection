import type {FireEvent} from "./fire"; import type {ModelInfo} from "./model";
export interface DashboardStats {totalEvents:number; highConfidence:number; wildfire:number; agriculture:number; industrial:number; mining:number; gas:number;}
export interface AnalyticsData {distribution:Record<string,number>; timeline:{date:string;events:number}[]; confidence:{band:string;count:number}[];}
export interface AnalysisResult {eventsProcessed:number; averageConfidence:number; highConfidence:number; predictionsReady:boolean;}
export type ApiEvent = FireEvent; export type ApiModel = ModelInfo;
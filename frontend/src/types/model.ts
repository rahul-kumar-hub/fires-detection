export interface ModelInfo {
 name:string; labelingFunctions:number; classes:number; events:number;
 config:Record<string,string|number>;
 metrics:null|Record<string,number>;
}
import type {AnalyticsData} from "../../types/api";
export const demoAnalytics:AnalyticsData={
 distribution:{Wildfire:109820,Agriculture:72140,Industrial:49380,Mining:36110,Gas:34620},
 timeline:["01","02","03","04","05","06","07","08","09"].map((date,i)=>({date:`Sep ${date}`,events:21000+i*1850-(i%3)*900})),
 confidence:[{band:"High",count:184230},{band:"Medium",count:76200},{band:"Low",count:41640}]
};
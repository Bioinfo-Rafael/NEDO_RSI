import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = new URL('.', import.meta.url).pathname.replace(/\/$/, '');
const data = JSON.parse(await fs.readFile(`${root}/outputs/excel_data.json`, 'utf8'));
const workbook = Workbook.create();
const letter = n => { let s = ''; for (n++; n; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + (n - 1) % 26) + s; return s; };
for (const [name, spec] of Object.entries(data)) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const end = letter(spec.columns.length - 1);
  const last = spec.rows.length + 1;
  sheet.getRange('A1').write([spec.columns, ...spec.rows]);
  sheet.getRange(`A1:${end}${last}`).format.font = {name:'Arial', size:10};
  sheet.getRange(`A1:${end}${last}`).format.verticalAlignment = 'top';
  sheet.getRange(`A1:${end}1`).format = {
    fill:'#27445C', font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    rowHeight:36, wrapText:true, horizontalAlignment:'center', verticalAlignment:'center'
  };
  for (let i=0; i<spec.columns.length; i++) {
    const col=letter(i), key=spec.columns[i];
    const numeric=['branching_factor','depth','total_branch_points','selected','discarded','completed'].includes(key);
    sheet.getRange(`${col}1:${col}${last}`).format.columnWidth = numeric ? 20 : /abstract|decision_context|search_pattern/.test(key) ? 70 : /raw/.test(key) ? 62 : 42;
    if (last>1) {
      const body=sheet.getRange(`${col}2:${col}${last}`);
      body.format.wrapText=true;
      body.format.horizontalAlignment=numeric?'right':'left';
      if(numeric)body.format.numberFormat='#,##0';
    }
  }
  if(last>1) {
    sheet.getRange(`A2:${end}${last}`).format.rowHeight=name==='Stats'?25:name==='Runs'?90:180;
    const table=sheet.tables.add(`A1:${end}${last}`,true,`${name}Records`);
    table.style='TableStyleMedium2';table.showFilterButton=true;
  }
  if(name!=='Stats') {sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(2);}
  if(name==='Branch_Points')sheet.tabColor='#27445C';
}
const stats=workbook.worksheets.getItem('Stats');
const last=data.Stats.rows.length+2;
stats.getRange(`A${last}`).values=[['TOTAL']];
for(const col of ['B','C','D','E'])stats.getRange(`${col}${last}`).formulas=[[`=SUM(${col}2:${col}${last-1})`]];
stats.getRange(`A${last}:E${last}`).format.font={name:'Arial',size:10,bold:true};
stats.getRange(`B${last}:E${last}`).format.numberFormat='#,##0';
stats.getRange(`G1:H5`).values=[['Dataset source','branch_memory.db'],['Cap (bytes)',5000000000],['Selection seed',17],['Raw full text','SQLite holds all selected raw text'],['Model','gpt-5.6-sol']];
stats.getRange('G1:G5').format.columnWidth=24;stats.getRange('H1:H5').format.columnWidth=48;
workbook.recalculate();
await fs.mkdir(`${root}/outputs/previews`,{recursive:true});
for(const [name,range] of [['Branch_Points','J1:K3'],['Runs','A1:D4'],['Stats',`A1:E${last}`]]) {
  const result=await workbook.inspect({kind:'table',sheetId:name,range,include:'values,formulas',tableMaxRows:8,tableMaxCols:5,maxChars:1800});
  console.log(result.ndjson);
  const preview=await workbook.render({sheetName:name,range,scale:1.3,format:'png'});
  await fs.writeFile(`${root}/outputs/previews/${name}.png`,new Uint8Array(await preview.arrayBuffer()));
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:30},summary:'Formula error scan'});
console.log(errors.ndjson);
const total=stats.getRange(`B${last}:E${last}`).values[0];
const expected=[0,0,0,0];for(const row of data.Stats.rows)for(let i=0;i<4;i++)expected[i]+=row[i+1];
if(JSON.stringify(total)!==JSON.stringify(expected))throw new Error(`Totals mismatch: ${JSON.stringify(total)}`);
const file=await SpreadsheetFile.exportXlsx(workbook);
await file.save(`${root}/branch_memory.xlsx`);
await fs.rename(`${root}/branch_memory.xlsx.inspect.ndjson`,`${root}/outputs/excel_inspect.ndjson`);
await fs.writeFile(`${root}/outputs/excel_verification.json`,JSON.stringify({sheets:Object.keys(data),rows:Object.fromEntries(Object.entries(data).map(([k,v])=>[k,v.rows.length])),totals:total,formulaErrors:errors.ndjson},null,2));
console.log('EXPORTED branch_memory.xlsx');

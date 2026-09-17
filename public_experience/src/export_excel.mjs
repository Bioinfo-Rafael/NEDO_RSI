import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const root=new URL('..',import.meta.url).pathname.replace(/\/$/,'');
const name=process.argv[2];
if (!['public_experience','combined_experience'].includes(name)) throw new Error('Expected workbook name');
const data=JSON.parse(await fs.readFile(`${root}/.excel_data/${name}.json`,'utf8'));
const wb=Workbook.create();
const letter=(n)=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
for(const [title,spec] of Object.entries(data.sheets)){
  const sheet=wb.worksheets.add(title);sheet.showGridLines=false;
  const lastCol=letter(spec.columns.length-1),lastRow=spec.rows.length+1;
  sheet.getRange('A1').write([spec.columns,...spec.rows]);
  const used=sheet.getRange(`A1:${lastCol}${lastRow}`);
  used.format.font={name:'Arial',size:10};used.format.verticalAlignment='top';
  sheet.getRange(`A1:${lastCol}1`).format={fill:'#234C65',font:{name:'Arial',size:10,color:'#FFFFFF',bold:true},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:32,wrapText:true};
  if(lastRow>1){
    sheet.getRange(`A2:${lastCol}${lastRow}`).format.rowHeight=62;
    const table=sheet.tables.add(`A1:${lastCol}${lastRow}`,true,`${title}Records`);table.style='TableStyleMedium2';table.showFilterButton=true;
  }
  for(let i=0;i<spec.columns.length;i++){
    const c=spec.columns[i],col=letter(i),range=sheet.getRange(`${col}1:${col}${lastRow}`);
    range.format.columnWidth=['depth','score','sequence_index','sequence_no','event_count'].includes(c)?14:c==='source_type'?19:c==='source_dataset'?48:c.endsWith('_id')?42:36;
    if(lastRow>1){sheet.getRange(`${col}2:${col}${lastRow}`).format.wrapText=true;
      if(c==='score')sheet.getRange(`${col}2:${col}${lastRow}`).format.numberFormat='0.########';
      else if(['depth','sequence_index','sequence_no','event_count'].includes(c))sheet.getRange(`${col}2:${col}${lastRow}`).format.numberFormat='#,##0';
    }
  }
  sheet.freezePanes.freezeRows(1);sheet.freezePanes.freezeColumns(2);
}
wb.recalculate();
await fs.mkdir(`${root}/reports/previews`,{recursive:true});
for(const title of ['Runs','Experiences','Raw_Events']){
  console.log((await wb.inspect({kind:'table',sheetId:title,range:'A1:H3',include:'values,formulas',tableMaxRows:3,tableMaxCols:8,maxChars:1500})).ndjson);
  const image=await wb.render({sheetName:title,range:'A1:D5',scale:1.2,format:'png'});
  await fs.writeFile(`${root}/reports/previews/${name}_${title}.png`,new Uint8Array(await image.arrayBuffer()));
}
console.log((await wb.inspect({kind:'formula',sheetId:'Experiences',range:'A1:AH5',maxChars:1000})).ndjson);
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(`${root}/${name}.xlsx`);
console.log('EXPORTED',name);

"""Verify exported XLSX values against the explicit run-complete sample."""
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from common import ROOT

NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def strings(element):
    value=''.join(t.text or '' for t in element.findall('.//m:t',NS))
    return re.sub(r'_x([0-9a-fA-F]{4})_',lambda m:chr(int(m[1],16)),value)

def main():
    results={}
    for name in ('public_experience','combined_experience'):
        expected=json.loads((ROOT/'.excel_data'/f'{name}.json').read_text())['sheets'];result={}
        with zipfile.ZipFile(ROOT/f'{name}.xlsx') as z:
            sheets=ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',NS)
            assert [s.attrib['name'] for s in sheets]==list(expected)
            shared=[strings(si) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS)] if 'xl/sharedStrings.xml' in z.namelist() else []
            for i,sheet in enumerate(sheets,1):
                title=sheet.attrib['name'];spec=expected[title]
                data=[spec['columns']]+spec['rows']
                tree=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
                rows=tree.findall('m:sheetData/m:row',NS)
                assert len(rows)==len(data),(name,title,'row count')
                assert not tree.findall('.//m:f',NS)
                count=0
                for r,row in enumerate(rows):
                    actual=[None]*len(spec['columns'])
                    for cell in row.findall('m:c',NS):
                        letters=re.match(r'[A-Z]+',cell.attrib['r'])[0];col=0
                        for letter in letters:col=col*26+ord(letter)-64
                        typ=cell.get('t');v=cell.find('m:v',NS)
                        assert typ!='e',(name,title,cell.attrib['r'])
                        if typ=='s':value=shared[int(v.text)]
                        elif typ=='inlineStr':value=strings(cell)
                        elif v is None:value=None
                        elif typ=='b':value=v.text=='1'
                        elif typ=='str':value=v.text or ''
                        else:value=float(v.text)
                        actual[col-1]=value
                    for col,(a,b) in enumerate(zip(actual,data[r])):
                        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                            assert math.isclose(a,b,rel_tol=1e-14,abs_tol=1e-14),(name,title,r,col,a,b)
                        else:
                            equal=a==b or (a in (None,'') and b in (None,'')) or (isinstance(b,str) and b.startswith("'=") and a==b[1:])
                            assert equal,(name,title,r,col,repr(a)[:100],repr(b)[:100])
                        count+=1
                result[title]={'rows_including_header':len(rows),'verified_values':count,'formula_count':0,'error_cells':0,'freeze_panes':tree.find('m:sheetViews/m:sheetView/m:pane',NS).attrib}
        results[name]=result
        print('verified',name,result,flush=True)
    (ROOT/'reports/excel_audit.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

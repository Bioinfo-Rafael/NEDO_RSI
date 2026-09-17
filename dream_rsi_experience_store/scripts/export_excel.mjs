import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = new URL("../..", import.meta.url).pathname.replace(/\/$/, "");
const data = JSON.parse(await fs.readFile(`${root}/experience_store/.excel_data.json`, "utf8"));
const output = `${root}/experience_store/experience.xlsx`;
const workbook = Workbook.create();
const font = "Arial";
const header = "#1F4E78";

function columnName(index) {
  let number = index + 1;
  let name = "";
  while (number) {
    number -= 1;
    name = String.fromCharCode(65 + (number % 26)) + name;
    number = Math.floor(number / 26);
  }
  return name;
}

const wide = new Set([
  "proposal", "prompt", "result_summary", "metrics_json", "artifacts_json",
  "commit_message", "source_reference", "source_files_json", "metadata_json",
  "command", "cwd", "stdout", "stderr", "message", "source_file", "raw_json",
  "display_path",
]);
const identifiers = new Set(["run_id", "node_id", "parent_id", "event_id", "commit_hash"]);

let tableIndex = 1;
for (const name of ["Runs", "Experiences", "Raw_Events"]) {
  const specification = data.sheets[name];
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const matrix = [specification.columns, ...specification.rows];
  sheet.getRange("A1").write(matrix);
  const lastColumn = columnName(specification.columns.length - 1);
  const lastRow = matrix.length;
  const used = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  used.format.font = { name: font, size: 10 };
  used.format.verticalAlignment = "top";
  used.format.wrapText = false;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: header,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 30;
  if (specification.rows.length) {
    const table = sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, `ReplayTable${tableIndex}`);
    table.style = "TableStyleMedium2";
    table.showFilterButton = true;
  }
  if (lastRow > 1) {
    const dataRows = sheet.getRange(`A2:${lastColumn}${lastRow}`);
    dataRows.format.rowHeight = name === "Runs" ? 54 : name === "Experiences" ? 42 : 30;
  }
  specification.columns.forEach((field, index) => {
    const column = columnName(index);
    sheet.getRange(`${column}:${column}`).format.columnWidth = wide.has(field)
      ? 38
      : identifiers.has(field)
        ? 24
        : 16;
    if (wide.has(field) && lastRow > 1) {
      sheet.getRange(`${column}2:${column}${lastRow}`).format.wrapText = true;
    }
    if (field === "score" && lastRow > 1) {
      sheet.getRange(`${column}2:${column}${lastRow}`).format.numberFormat = "0.000000";
    }
    if ((field === "timestamp" || field.endsWith("_time")) && lastRow > 1) {
      sheet.getRange(`${column}2:${column}${lastRow}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
    }
    if ((field.endsWith("_count") || field.endsWith("_index") || field === "depth" || field === "sequence_no") && lastRow > 1) {
      sheet.getRange(`${column}2:${column}${lastRow}`).format.numberFormat = "#,##0";
    }
  });
  sheet.freezePanes.freezeRows(1);
  if (name === "Experiences") sheet.freezePanes.freezeColumns(3);
  if (name === "Raw_Events") sheet.freezePanes.freezeColumns(2);
  tableIndex += 1;
}

workbook.recalculate();
for (const [name, range] of [
  ["Runs", "A1:N4"],
  ["Experiences", "A1:P12"],
  ["Raw_Events", "A1:S12"],
]) {
  const preview = await workbook.render({ sheetName: name, range, scale: 1.2, format: "png" });
  await fs.writeFile(
    `${root}/experience_store/.excel_preview_${name}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
console.log((await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 3000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "table",
  sheetId: "Experiences",
  range: "A1:P8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 16,
  maxChars: 8000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "formula error scan",
})).ndjson);
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(output);
console.log(output);

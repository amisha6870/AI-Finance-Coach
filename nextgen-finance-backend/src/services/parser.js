const fs = require("fs");
const csv = require("csv-parser");

function parseCSV(filePath) {
  return new Promise((resolve, reject) => {
    const results = [];

    fs.createReadStream(filePath)
      .pipe(csv({
        mapHeaders: ({ header }) => String(header || '').replace(/^\uFEFF/, '').trim(),
        mapValues: ({ value }) => typeof value === 'string' ? value.trim() : value,
      }))
      .on("data", (data) => results.push(data))
      .on("end", () => resolve(results))
      .on("error", reject);
  });
}

module.exports = parseCSV;

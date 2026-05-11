async function run() {
  try {
    const { trainMlModels } = require('../services/mlInsightsService');
    const result = await trainMlModels();
    process.stdout.write(`${JSON.stringify(result)}\n`);
  } catch (error) {
    const payload = {
      error: error.message,
      failures: error.failures || [],
      attemptedCommands: error.attemptedCommands || [],
    };
    process.stderr.write(`${JSON.stringify(payload)}\n`);
    process.exit(1);
  }
}

run();

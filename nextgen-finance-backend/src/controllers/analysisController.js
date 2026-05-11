const { successResponse } = require('../utils/response');
const { getMlInsightsForUser, trainMlModels } = require('../services/mlInsightsService');

const getMyAnalysis = async (req, res, next) => {
  try {
    const result = await getMlInsightsForUser(req.user._id);

    return successResponse(res, 'ML financial analysis generated successfully', {
      monthlyDataset: result.monthlyRows,
      currentFeatures: result.currentRow,
      analytics: result.analytics || {},
      ml: result.ml,
      predictionSummary: {
        source: result.ml?.source || 'fallback_rules',
        overspending_risk: result.ml?.overspending_risk || result.ml?.overspending?.riskLevel || 'Low',
        confidence: result.ml?.confidence || result.ml?.overspending?.confidence || 60,
        predicted_expense: result.ml?.predicted_expense || result.ml?.trend?.nextMonthExpense || result.currentRow?.expenses || 0,
        spender_type: result.ml?.spender_type || result.ml?.behavior?.segment || 'Unknown',
        monthly_health_score: result.currentRow?.financialHealthScore || result.ml?.analytics?.monthlyHealthScore || 0,
        spending_volatility: result.currentRow?.spendingVolatility || result.ml?.analytics?.spendingVolatility || 0,
        anomaly_detected: Boolean(result.currentRow?.anomaly || result.ml?.analytics?.anomaly),
        anomaly_score: result.currentRow?.anomalyScore || result.ml?.analytics?.anomalyScore || 0,
      },
    });
  } catch (error) {
    return next(error);
  }
};

const retrainModels = async (req, res, next) => {
  try {
    const result = await trainMlModels();
    return successResponse(res, 'ML models trained successfully', {
      ...result,
      source: 'trained_model',
    });
  } catch (error) {
    return next(error);
  }
};

module.exports = {
  getMyAnalysis,
  retrainModels,
};

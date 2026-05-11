const express = require('express');
const { protect } = require('../middleware/authMiddleware');
const { getMyAnalysis } = require('../controllers/analysisController');

const router = express.Router();

router.use(protect);

router.get('/me', getMyAnalysis);

module.exports = router;

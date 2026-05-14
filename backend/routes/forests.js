const express = require('express');
const router = express.Router();
const forestsController = require('../controllers/forestsController');

router.get('/', forestsController.getAllForests);
router.get('/:id', forestsController.getForestById);
router.post('/', forestsController.createForest);
router.put('/:id', forestsController.updateForest);
router.delete('/:id', forestsController.deleteForest);

module.exports = router;
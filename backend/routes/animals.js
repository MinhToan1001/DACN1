const express = require('express');
const router = express.Router();
const animalsController = require('../controllers/animalsController');

router.get('/', animalsController.getAllAnimals);
router.get('/:id', animalsController.getAnimalById);
router.post('/', animalsController.createAnimal);
router.put('/:id', animalsController.updateAnimal);
router.delete('/:id', animalsController.deleteAnimal);

module.exports = router;

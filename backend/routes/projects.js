const express = require('express');
const router = express.Router();
const projectsController = require('../controllers/projectsController');

router.get("/", projectsController.getAllProjects);
router.get("/:id", projectsController.getProjectById);
router.post("/", projectsController.createProject);
router.put("/:id", projectsController.updateProject);
router.delete("/:id", projectsController.deleteProject);

module.exports = router;
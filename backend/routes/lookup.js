const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data'); // Required for sending files with axios

// Multer storage configuration for handling image uploads
const storage = multer.memoryStorage(); // Store the file in memory
const upload = multer({ storage: storage });

// --- API Endpoint for Animal Classification ---
// This endpoint will receive an image from the frontend and forward it to the Python Flask API.
router.post('/classify', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image file uploaded.' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
        });

        // Make a request to the Python Flask API
        const pythonResponse = await axios.post('http://localhost:5000/predict', formData, {
            headers: {
                ...formData.getHeaders(),
                'Content-Length': formData.getLengthSync(),
            },
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
        });

        // Forward the Python API's response to the client
        res.json(pythonResponse.data);

    } catch (error) {
        console.error('Error forwarding classification request to Python:', error.message);
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error('Python API Response Error Data:', error.response.data);
            return res.status(error.response.status).json(error.response.data);
        } else if (error.request) {
            // The request was made but no response was received
            console.error('Python API No Response:', error.request);
            return res.status(500).json({ error: 'No response from Python classification service.' });
        } else {
            // Something happened in setting up the request that triggered an Error
            console.error('Python API Request Setup Error:', error.message);
            return res.status(500).json({ error: 'Error setting up request to Python classification service.' });
        }
    }
});

// --- API Endpoint for Chatbot ---
// This endpoint will receive a message from the frontend and forward it to the Python Flask API.
router.post('/chat', async (req, res) => {
    const { message } = req.body;

    if (!message) {
        return res.status(400).json({ error: 'No message provided.' });
    }

    try {
        // Make a request to the Python Flask API
        const pythonResponse = await axios.post('http://localhost:5000/chat', { message });

        // Forward the Python API's response to the client
        res.json(pythonResponse.data);

    } catch (error) {
        console.error('Error forwarding chat request to Python:', error.message);
        if (error.response) {
            console.error('Python API Response Error Data:', error.response.data);
            return res.status(error.response.status).json(error.response.data);
        } else if (error.request) {
            console.error('Python API No Response:', error.request);
            return res.status(500).json({ error: 'No response from Python chat service.' });
        } else {
            console.error('Python API Request Setup Error:', error.message);
            return res.status(500).json({ error: 'Error setting up request to Python chat service.' });
        }
    }
});

module.exports = router;

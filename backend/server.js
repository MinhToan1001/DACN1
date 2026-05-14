const express = require('express')
const cors = require('cors')

const app = express()
app.use(cors())
app.use(express.json())
app.use(express.urlencoded({ extended: true })); // Nếu gửi form-data

//API test
app.get('/', (req, res) => {
  res.send('API đang hoạt động!')
});

//Chạy server
const PORT = process.env.PORT || 5001
app.listen(PORT, () => {
  console.log(`🚀 Server đang chạy tại http://localhost:${PORT}`)
});

// Import routes
const animalsRoutes = require('./routes/animals')
const newsRoutes = require('./routes/news')
const projectsRoutes = require('./routes/projects')
const forestsRoutes = require('./routes/forests')
const contactRoutes = require('./routes/contact')
const authRoutes = require("./routes/auth");
const paymentRoutes = require('./routes/payment');


// Sử dụng routes
app.use('/api/animals', animalsRoutes)
app.use('/api/news', newsRoutes)
app.use('/api/projects', projectsRoutes)
app.use('/api/contact', contactRoutes)
app.use('/api/forests-map', forestsRoutes)
app.use("/api/auth", authRoutes);
app.use('/api/payment', paymentRoutes);
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const path = require('path');
require('dotenv').config();

// Import routes
const authRoutes = require('./routes/authRoutes');
const userRoutes = require('./routes/userRoutes');
const detectionRoutes = require('./routes/detectionRoutes');
const uploadRoutes = require('./routes/uploadRoutes');
const streamRoutes = require('./routes/streamRoutes');

// Import middleware
const { errorHandler } = require('./middleware/errorHandler');
const { authenticateToken } = require('./middleware/auth');

// Initialize data service
const dataService = require('./services/dataService');

const app = express();
const PORT = process.env.PORT || 5002;

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Don't exit on SIGINT for now - comment this out for testing
// process.on('SIGINT', () => {
//   console.log('Server shutting down gracefully...');
//   process.exit(0);
// });

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use(limiter);

// Logging
app.use(morgan('combined'));

// Body parsing middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve uploaded files
app.use('/uploads', express.static('uploads'));
// Serve processed outputs (maps to project-root/outputs_vid)
const rootDir = path.resolve(__dirname, '../../..');
const outputsDir = path.join(rootDir, 'outputs_vid');
app.use('/outputs', express.static(outputsDir));

// Initialize data storage
console.log('Initializing local data storage...');

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/users', authenticateToken, userRoutes);
app.use('/api/detection', authenticateToken, detectionRoutes);
app.use('/api/upload', authenticateToken, uploadRoutes);
app.use('/api/stream', streamRoutes);

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'Vehicle Occlusion API is running',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    storage: 'Local file system'
  });
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Vehicle Occlusion Detection API',
    version: '1.0.0',
    documentation: '/api/docs',
    endpoints: {
      auth: '/api/auth',
      users: '/api/users',
      detection: '/api/detection',
      upload: '/api/upload',
      health: '/api/health'
    }
  });
});

// Error handling middleware
app.use(errorHandler);

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Endpoint not found'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log('Using local file storage for data persistence');
});

module.exports = app;
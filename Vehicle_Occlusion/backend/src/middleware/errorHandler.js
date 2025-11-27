const errorHandler = (err, req, res, next) => {
  let error = { ...err };
  error.message = err.message;

  // Log error for debugging
  console.error('Error Handler:', err);

  // Mongoose bad ObjectId
  if (err.name === 'CastError') {
    const message = 'Resource not found';
    error = {
      message,
      status: 404
    };
  }

  // Mongoose duplicate key
  if (err.code === 11000) {
    const message = 'Duplicate field value entered';
    const field = Object.keys(err.keyValue)[0];
    error = {
      message: `${field.charAt(0).toUpperCase() + field.slice(1)} already exists`,
      status: 400,
      field
    };
  }

  // Mongoose validation error
  if (err.name === 'ValidationError') {
    const message = Object.values(err.errors).map(val => val.message).join(', ');
    error = {
      message,
      status: 400,
      errors: err.errors
    };
  }

  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    error = {
      message: 'Invalid token',
      status: 401
    };
  }

  if (err.name === 'TokenExpiredError') {
    error = {
      message: 'Token expired',
      status: 401
    };
  }

  // Multer errors (file upload)
  if (err.code === 'LIMIT_FILE_SIZE') {
    error = {
      message: 'File size too large',
      status: 400
    };
  }

  if (err.code === 'LIMIT_FILE_COUNT') {
    error = {
      message: 'Too many files uploaded',
      status: 400
    };
  }

  if (err.code === 'LIMIT_UNEXPECTED_FILE') {
    error = {
      message: 'Unexpected file field',
      status: 400
    };
  }

  // File type errors
  if (err.message && err.message.includes('Invalid file type')) {
    error = {
      message: 'Invalid file type. Only images and videos are allowed',
      status: 400
    };
  }

  // Database connection errors
  if (err.name === 'MongoError' || err.name === 'MongooseError') {
    error = {
      message: 'Database connection error',
      status: 500
    };
  }

  // Network timeout errors
  if (err.code === 'ECONNABORTED' || err.code === 'ETIMEDOUT') {
    error = {
      message: 'Request timeout',
      status: 408
    };
  }

  // Rate limit errors
  if (err.status === 429) {
    error = {
      message: 'Too many requests. Please try again later.',
      status: 429,
      retryAfter: err.retryAfter || 60
    };
  }

  // Default to 500 server error
  const status = error.status || err.statusCode || 500;
  const message = error.message || 'Server Error';

  const response = {
    success: false,
    message,
    ...(process.env.NODE_ENV === 'development' && {
      stack: err.stack,
      error: err
    })
  };

  // Add additional error details if available
  if (error.field) response.field = error.field;
  if (error.errors) response.errors = error.errors;
  if (error.retryAfter) response.retryAfter = error.retryAfter;

  res.status(status).json(response);
};

// 404 handler
const notFound = (req, res, next) => {
  const error = new Error(`Not found - ${req.originalUrl}`);
  error.status = 404;
  next(error);
};

// Async error handler wrapper
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// Custom error class
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.status = `${statusCode}`.startsWith('4') ? 'fail' : 'error';
    this.isOperational = true;

    Error.captureStackTrace(this, this.constructor);
  }
}

module.exports = {
  errorHandler,
  notFound,
  asyncHandler,
  AppError
};
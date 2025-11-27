const express = require('express');
const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const Upload = require('../models/Upload');
const { asyncHandler } = require('../middleware/errorHandler');

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}-${Date.now()}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const fileFilter = (req, file, cb) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg', 'video/mp4', 'video/avi'];
  
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only images (JPEG, PNG) and videos (MP4, AVI) are allowed'), false);
  }
};

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
    files: 5 // Maximum 5 files
  },
  fileFilter: fileFilter
});

// @route   POST /api/upload/single
// @desc    Upload a single file
// @access  Private
router.post('/single', upload.single('file'), asyncHandler(async (req, res) => {
  if (!req.file) {
    return res.status(400).json({
      success: false,
      message: 'No file uploaded'
    });
  }

  // Create upload record
  const uploadRecord = new Upload({
    user: req.user._id,
    originalName: req.file.originalname,
    filename: req.file.filename,
    mimetype: req.file.mimetype,
    size: req.file.size,
    path: req.file.path,
    url: `/uploads/${req.file.filename}`,
    checksum: req.file.filename // Simple checksum for now
  });

  await uploadRecord.save();

  // Update user stats
  await req.user.incrementUploadStats();

  res.status(201).json({
    success: true,
    message: 'File uploaded successfully',
    data: {
      upload: uploadRecord
    }
  });
}));

// @route   POST /api/upload/multiple
// @desc    Upload multiple files
// @access  Private
router.post('/multiple', upload.array('files', 5), asyncHandler(async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({
      success: false,
      message: 'No files uploaded'
    });
  }

  const uploadRecords = [];

  for (const file of req.files) {
    const uploadRecord = new Upload({
      user: req.user._id,
      originalName: file.originalname,
      filename: file.filename,
      mimetype: file.mimetype,
      size: file.size,
      path: file.path,
      url: `/uploads/${file.filename}`,
      checksum: file.filename
    });

    await uploadRecord.save();
    uploadRecords.push(uploadRecord);
  }

  // Update user stats
  await req.user.incrementUploadStats();

  res.status(201).json({
    success: true,
    message: `${req.files.length} files uploaded successfully`,
    data: {
      uploads: uploadRecords
    }
  });
}));

// @route   GET /api/upload/history
// @desc    Get user's upload history
// @access  Private
router.get('/history', asyncHandler(async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const skip = (page - 1) * limit;

  const uploads = await Upload.find({ user: req.user._id })
    .sort({ createdAt: -1 })
    .skip(skip)
    .limit(limit)
    .populate('user', 'username email');

  const total = await Upload.countDocuments({ user: req.user._id });

  res.json({
    success: true,
    data: {
      uploads,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    }
  });
}));

// @route   GET /api/upload/:id
// @desc    Get specific upload details
// @access  Private
router.get('/:id', asyncHandler(async (req, res) => {
  const upload = await Upload.findOne({
    _id: req.params.id,
    user: req.user._id
  });

  if (!upload) {
    return res.status(404).json({
      success: false,
      message: 'Upload not found'
    });
  }

  res.json({
    success: true,
    data: {
      upload
    }
  });
}));

// @route   DELETE /api/upload/:id
// @desc    Delete an upload
// @access  Private
router.delete('/:id', asyncHandler(async (req, res) => {
  const upload = await Upload.findOne({
    _id: req.params.id,
    user: req.user._id
  });

  if (!upload) {
    return res.status(404).json({
      success: false,
      message: 'Upload not found'
    });
  }

  // Delete file from filesystem
  const fs = require('fs').promises;
  try {
    await fs.unlink(upload.path);
  } catch (error) {
    console.error('Error deleting file:', error);
  }

  // Delete from database
  await Upload.findByIdAndDelete(req.params.id);

  res.json({
    success: true,
    message: 'Upload deleted successfully'
  });
}));

module.exports = router;
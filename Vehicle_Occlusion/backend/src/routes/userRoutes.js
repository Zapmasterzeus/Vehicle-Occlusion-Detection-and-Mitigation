const express = require('express');
const Joi = require('joi');
const User = require('../models/User');
const { asyncHandler } = require('../middleware/errorHandler');
const { authorizeRoles } = require('../middleware/auth');

const router = express.Router();

// @route   GET /api/users/profile
// @desc    Get current user profile
// @access  Private
router.get('/profile', asyncHandler(async (req, res) => {
  res.json({
    success: true,
    data: {
      user: req.user
    }
  });
}));

// @route   PUT /api/users/profile
// @desc    Update user profile
// @access  Private
router.put('/profile', asyncHandler(async (req, res) => {
  const updateSchema = Joi.object({
    firstName: Joi.string().max(50),
    lastName: Joi.string().max(50),
    preferences: Joi.object({
      theme: Joi.string().valid('light', 'dark'),
      notifications: Joi.object({
        email: Joi.boolean(),
        push: Joi.boolean()
      }),
      language: Joi.string()
    })
  });

  const { error } = updateSchema.validate(req.body);
  if (error) {
    return res.status(400).json({
      success: false,
      message: error.details[0].message
    });
  }

  const allowedUpdates = ['firstName', 'lastName', 'preferences'];
  const updates = {};
  
  for (const key of allowedUpdates) {
    if (req.body[key] !== undefined) {
      updates[key] = req.body[key];
    }
  }

  const user = await User.findByIdAndUpdate(
    req.user._id,
    updates,
    { new: true, runValidators: true }
  );

  res.json({
    success: true,
    message: 'Profile updated successfully',
    data: {
      user
    }
  });
}));

// @route   GET /api/users/stats
// @desc    Get user statistics
// @access  Private
router.get('/stats', asyncHandler(async (req, res) => {
  const Detection = require('../models/Detection');
  const Upload = require('../models/Upload');

  const [totalDetections, totalUploads, recentDetections] = await Promise.all([
    Detection.countDocuments({ user: req.user._id }),
    Upload.countDocuments({ user: req.user._id }),
    Detection.find({ user: req.user._id })
      .sort({ createdAt: -1 })
      .limit(5)
      .populate('uploadedFile', 'originalName')
  ]);

  const stats = {
    totalDetections,
    totalUploads,
    recentDetections,
    userStats: req.user.stats
  };

  res.json({
    success: true,
    data: {
      stats
    }
  });
}));

// Admin only routes
// @route   GET /api/users
// @desc    Get all users (admin only)
// @access  Private/Admin
router.get('/', authorizeRoles('admin'), asyncHandler(async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const skip = (page - 1) * limit;

  const users = await User.find()
    .select('-password')
    .sort({ createdAt: -1 })
    .skip(skip)
    .limit(limit);

  const total = await User.countDocuments();

  res.json({
    success: true,
    data: {
      users,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    }
  });
}));

// @route   PUT /api/users/:id/role
// @desc    Update user role (admin only)
// @access  Private/Admin
router.put('/:id/role', authorizeRoles('admin'), asyncHandler(async (req, res) => {
  const { role } = req.body;
  
  if (!['user', 'admin', 'analyst'].includes(role)) {
    return res.status(400).json({
      success: false,
      message: 'Invalid role'
    });
  }

  const user = await User.findByIdAndUpdate(
    req.params.id,
    { role },
    { new: true }
  ).select('-password');

  if (!user) {
    return res.status(404).json({
      success: false,
      message: 'User not found'
    });
  }

  res.json({
    success: true,
    message: 'User role updated successfully',
    data: {
      user
    }
  });
}));

// @route   PUT /api/users/:id/status
// @desc    Activate/deactivate user (admin only)
// @access  Private/Admin
router.put('/:id/status', authorizeRoles('admin'), asyncHandler(async (req, res) => {
  const { isActive } = req.body;
  
  if (typeof isActive !== 'boolean') {
    return res.status(400).json({
      success: false,
      message: 'isActive must be a boolean'
    });
  }

  const user = await User.findByIdAndUpdate(
    req.params.id,
    { isActive },
    { new: true }
  ).select('-password');

  if (!user) {
    return res.status(404).json({
      success: false,
      message: 'User not found'
    });
  }

  res.json({
    success: true,
    message: `User ${isActive ? 'activated' : 'deactivated'} successfully`,
    data: {
      user
    }
  });
}));

module.exports = router;
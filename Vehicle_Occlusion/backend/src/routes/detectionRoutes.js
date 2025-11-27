const express = require('express');
const Detection = require('../models/Detection');
const Upload = require('../models/Upload');
const { asyncHandler } = require('../middleware/errorHandler');

const router = express.Router();

// @route   POST /api/detection/analyze
// @desc    Start vehicle detection analysis
// @access  Private
router.post('/analyze', asyncHandler(async (req, res) => {
  const { uploadId, parameters } = req.body;

  if (!uploadId) {
    return res.status(400).json({
      success: false,
      message: 'Upload ID is required'
    });
  }

  // Verify upload exists and belongs to user
  const upload = await Upload.findOne({
    _id: uploadId,
    user: req.user._id
  });

  if (!upload) {
    return res.status(404).json({
      success: false,
      message: 'Upload not found'
    });
  }

  // Check if detection already exists for this upload
  const existingDetection = await Detection.findOne({ uploadedFile: uploadId });
  if (existingDetection) {
    return res.status(400).json({
      success: false,
      message: 'Detection already exists for this upload',
      data: { detectionId: existingDetection._id }
    });
  }

  // Create new detection record
  const detection = new Detection({
    user: req.user._id,
    uploadedFile: uploadId,
    status: 'pending',
    results: {
      processingMetadata: {
        parameters: parameters || {}
      }
    }
  });

  await detection.save();

  // Start processing (mock implementation)
  // In a real application, this would trigger ML processing
  setTimeout(async () => {
    try {
      await processVehicleDetection(detection._id);
    } catch (error) {
      console.error('Detection processing error:', error);
    }
  }, 100);

  res.status(201).json({
    success: true,
    message: 'Detection analysis started',
    data: {
      detectionId: detection._id,
      status: detection.status
    }
  });
}));

// @route   GET /api/detection/:id
// @desc    Get detection results
// @access  Private
router.get('/:id', asyncHandler(async (req, res) => {
  const detection = await Detection.findOne({
    _id: req.params.id,
    user: req.user._id
  }).populate('uploadedFile', 'originalName filename url metadata');

  if (!detection) {
    return res.status(404).json({
      success: false,
      message: 'Detection not found'
    });
  }

  res.json({
    success: true,
    data: {
      detection
    }
  });
}));

// @route   GET /api/detection/status/:id
// @desc    Get detection status
// @access  Private
router.get('/status/:id', asyncHandler(async (req, res) => {
  const detection = await Detection.findOne({
    _id: req.params.id,
    user: req.user._id
  }).select('status processingStartTime processingEndTime processingDuration');

  if (!detection) {
    return res.status(404).json({
      success: false,
      message: 'Detection not found'
    });
  }

  res.json({
    success: true,
    data: {
      id: detection._id,
      status: detection.status,
      processingStartTime: detection.processingStartTime,
      processingEndTime: detection.processingEndTime,
      processingDuration: detection.processingDuration
    }
  });
}));

// @route   GET /api/detection
// @desc    Get user's detection history
// @access  Private
router.get('/', asyncHandler(async (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const skip = (page - 1) * limit;
  const status = req.query.status;

  const query = { user: req.user._id };
  if (status) {
    query.status = status;
  }

  const detections = await Detection.find(query)
    .populate('uploadedFile', 'originalName filename url')
    .sort({ createdAt: -1 })
    .skip(skip)
    .limit(limit);

  const total = await Detection.countDocuments(query);

  res.json({
    success: true,
    data: {
      detections,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    }
  });
}));

// @route   DELETE /api/detection/:id
// @desc    Delete detection record
// @access  Private
router.delete('/:id', asyncHandler(async (req, res) => {
  const detection = await Detection.findOne({
    _id: req.params.id,
    user: req.user._id
  });

  if (!detection) {
    return res.status(404).json({
      success: false,
      message: 'Detection not found'
    });
  }

  await Detection.findByIdAndDelete(req.params.id);

  res.json({
    success: true,
    message: 'Detection deleted successfully'
  });
}));

// @route   POST /api/detection/:id/annotate
// @desc    Add annotation to detection
// @access  Private
router.post('/:id/annotate', asyncHandler(async (req, res) => {
  const { vehicleId, correctedType, correctedOcclusion, notes } = req.body;

  const detection = await Detection.findOne({
    _id: req.params.id,
    user: req.user._id
  });

  if (!detection) {
    return res.status(404).json({
      success: false,
      message: 'Detection not found'
    });
  }

  const annotation = {
    user: req.user._id,
    vehicleId,
    correctedType,
    correctedOcclusion,
    notes
  };

  detection.annotations.push(annotation);
  await detection.save();

  res.json({
    success: true,
    message: 'Annotation added successfully',
    data: {
      annotation
    }
  });
}));

// Mock ML processing function
async function processVehicleDetection(detectionId) {
  try {
    const detection = await Detection.findById(detectionId);
    if (!detection) return;

    // Update status to processing
    detection.status = 'processing';
    detection.processingStartTime = new Date();
    await detection.save();

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Mock detection results
    const mockResults = {
      totalVehicles: 3,
      occludedVehicles: 1,
      occlusionPercentage: 33.33,
      vehicles: [
        {
          id: 'vehicle_1',
          type: 'car',
          confidence: 0.95,
          boundingBox: { x: 100, y: 150, width: 200, height: 150 },
          occlusion: {
            isOccluded: false,
            occlusionLevel: 'none',
            occlusionPercentage: 0
          },
          features: {
            color: 'blue',
            size: 'medium',
            orientation: 0
          }
        },
        {
          id: 'vehicle_2',
          type: 'truck',
          confidence: 0.88,
          boundingBox: { x: 350, y: 120, width: 300, height: 200 },
          occlusion: {
            isOccluded: true,
            occlusionLevel: 'partial',
            occlusionPercentage: 35,
            occludedBy: ['vehicle_3']
          },
          features: {
            color: 'white',
            size: 'large',
            orientation: 15
          }
        },
        {
          id: 'vehicle_3',
          type: 'car',
          confidence: 0.92,
          boundingBox: { x: 500, y: 180, width: 180, height: 140 },
          occlusion: {
            isOccluded: false,
            occlusionLevel: 'none',
            occlusionPercentage: 0
          },
          features: {
            color: 'red',
            size: 'medium',
            orientation: -10
          }
        }
      ],
      processingMetadata: {
        modelVersion: 'v1.0.0',
        algorithm: 'YOLO + Custom Occlusion Analysis',
        computeTime: 4850,
        memoryUsage: 1024
      }
    };

    // Update detection with results
    detection.status = 'completed';
    detection.processingEndTime = new Date();
    detection.results = { ...detection.results, ...mockResults };
    
    await detection.save();

    console.log(`Detection ${detectionId} completed successfully`);
  } catch (error) {
    console.error(`Detection ${detectionId} failed:`, error);
    
    // Update detection with error
    const detection = await Detection.findById(detectionId);
    if (detection) {
      detection.status = 'failed';
      detection.processingEndTime = new Date();
      detection.errorDetails = {
        code: 'PROCESSING_ERROR',
        message: error.message
      };
      await detection.save();
    }
  }
}

module.exports = router;
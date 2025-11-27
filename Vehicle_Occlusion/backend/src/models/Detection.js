const dataService = require('../services/dataService');

class Detection {
  constructor(detectionData) {
    this.id = detectionData.id;
    this.userId = detectionData.userId;
    this.uploadId = detectionData.uploadId;
    this.status = detectionData.status || 'pending';
    this.processingStartTime = detectionData.processingStartTime;
    this.processingEndTime = detectionData.processingEndTime;
    this.processingDuration = detectionData.processingDuration;
    this.results = detectionData.results || {
      totalVehicles: 0,
      occludedVehicles: 0,
      occlusionPercentage: 0,
      vehicles: [],
      imageMetadata: {},
      processingMetadata: {}
    };
    this.annotations = detectionData.annotations || [];
    this.errorDetails = detectionData.errorDetails || {};
    this.metrics = detectionData.metrics || {};
    this.createdAt = detectionData.createdAt;
    this.updatedAt = detectionData.updatedAt;
  }

  // Virtual properties
  get processingTimeFormatted() {
    if (!this.processingDuration) return null;
    
    const seconds = Math.floor(this.processingDuration / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  get occlusionSummary() {
    if (!this.results || !this.results.vehicles) return null;
    
    const vehicles = this.results.vehicles;
    const summary = {
      none: 0,
      partial: 0,
      heavy: 0,
      complete: 0
    };
    
    vehicles.forEach(vehicle => {
      if (vehicle.occlusion && vehicle.occlusion.occlusionLevel) {
        summary[vehicle.occlusion.occlusionLevel]++;
      }
    });
    
    return summary;
  }

  // Static methods
  static async findById(id) {
    const detectionData = await dataService.getDetectionById(id);
    return detectionData ? new Detection(detectionData) : null;
  }

  static async findByUserId(userId) {
    const detectionsData = await dataService.getDetectionsByUserId(userId);
    return detectionsData.map(detectionData => new Detection(detectionData));
  }

  static async findByUploadId(uploadId) {
    const detectionsData = await dataService.getDetectionsByUploadId(uploadId);
    return detectionsData.map(detectionData => new Detection(detectionData));
  }

  static async findAll() {
    const detectionsData = await dataService.getDetections();
    return detectionsData.map(detectionData => new Detection(detectionData));
  }

  static async create(detectionData) {
    const newDetectionData = await dataService.createDetection(detectionData);
    return new Detection(newDetectionData);
  }

  // Instance methods
  async save() {
    // Calculate processing duration if both times are set
    if (this.processingStartTime && this.processingEndTime) {
      this.processingDuration = new Date(this.processingEndTime) - new Date(this.processingStartTime);
    }
    
    // Calculate occlusion percentage
    if (this.results && this.results.vehicles && this.results.vehicles.length > 0) {
      const occludedCount = this.results.vehicles.filter(v => 
        v.occlusion && v.occlusion.isOccluded
      ).length;
      
      this.results.occludedVehicles = occludedCount;
      this.results.occlusionPercentage = (occludedCount / this.results.vehicles.length) * 100;
    }

    const detectionData = {
      userId: this.userId,
      uploadId: this.uploadId,
      status: this.status,
      processingStartTime: this.processingStartTime,
      processingEndTime: this.processingEndTime,
      processingDuration: this.processingDuration,
      results: this.results,
      annotations: this.annotations,
      errorDetails: this.errorDetails,
      metrics: this.metrics
    };

    if (this.id) {
      const updatedData = await dataService.updateDetection(this.id, detectionData);
      if (updatedData) {
        Object.assign(this, updatedData);
      }
    } else {
      const newDetectionData = await dataService.createDetection(detectionData);
      Object.assign(this, newDetectionData);
    }
    
    return this;
  }

  async remove() {
    if (this.id) {
      return await dataService.deleteDetection(this.id);
    }
    return false;
  }

  async addAnnotation(annotation) {
    annotation.timestamp = new Date().toISOString();
    this.annotations.push(annotation);
    return await this.save();
  }

  async updateStatus(status, errorDetails = null) {
    this.status = status;
    if (status === 'processing' && !this.processingStartTime) {
      this.processingStartTime = new Date().toISOString();
    } else if (['completed', 'failed', 'cancelled'].includes(status) && !this.processingEndTime) {
      this.processingEndTime = new Date().toISOString();
    }
    
    if (errorDetails) {
      this.errorDetails = errorDetails;
    }
    
    return await this.save();
  }

  // Utility methods
  static createSampleDetection(userId, uploadId) {
    return {
      userId,
      uploadId,
      status: 'completed',
      processingStartTime: new Date(Date.now() - 5000).toISOString(),
      processingEndTime: new Date().toISOString(),
      results: {
        totalVehicles: 3,
        occludedVehicles: 1,
        occlusionPercentage: 33.33,
        vehicles: [
          {
            id: 'v1',
            type: 'car',
            confidence: 0.95,
            boundingBox: { x: 100, y: 150, width: 200, height: 120 },
            occlusion: {
              isOccluded: false,
              occlusionLevel: 'none',
              occlusionPercentage: 0,
              occludedBy: []
            },
            features: {
              color: 'blue',
              size: 'medium',
              orientation: 0
            }
          },
          {
            id: 'v2',
            type: 'truck',
            confidence: 0.88,
            boundingBox: { x: 350, y: 100, width: 180, height: 140 },
            occlusion: {
              isOccluded: true,
              occlusionLevel: 'partial',
              occlusionPercentage: 25,
              occludedBy: ['v3']
            },
            features: {
              color: 'red',
              size: 'large',
              orientation: 15
            }
          },
          {
            id: 'v3',
            type: 'car',
            confidence: 0.92,
            boundingBox: { x: 320, y: 180, width: 190, height: 110 },
            occlusion: {
              isOccluded: false,
              occlusionLevel: 'none',
              occlusionPercentage: 0,
              occludedBy: []
            },
            features: {
              color: 'white',
              size: 'medium',
              orientation: -10
            }
          }
        ],
        imageMetadata: {
          originalWidth: 1920,
          originalHeight: 1080,
          processedWidth: 1920,
          processedHeight: 1080,
          format: 'JPEG',
          size: 2048576
        },
        processingMetadata: {
          modelVersion: '1.0.0',
          algorithm: 'YOLO-v8',
          parameters: {
            threshold: 0.5,
            nms_threshold: 0.4
          },
          computeTime: 1250,
          memoryUsage: 512
        }
      }
    };
  }
}

module.exports = Detection;
const dataService = require('../services/dataService');

class Upload {
  constructor(uploadData) {
    this.id = uploadData.id;
    this.userId = uploadData.userId;
    this.originalName = uploadData.originalName;
    this.filename = uploadData.filename;
    this.mimetype = uploadData.mimetype;
    this.size = uploadData.size;
    this.path = uploadData.path;
    this.url = uploadData.url;
    this.fileType = uploadData.fileType;
    this.metadata = uploadData.metadata || {};
    this.thumbnailPath = uploadData.thumbnailPath || null;
    this.thumbnailUrl = uploadData.thumbnailUrl || null;
    this.status = uploadData.status || 'uploaded';
    this.tags = uploadData.tags || [];
    this.description = uploadData.description || '';
    this.isPublic = uploadData.isPublic || false;
    this.uploadSource = uploadData.uploadSource || 'web';
    this.processingHistory = uploadData.processingHistory || [];
    this.downloadCount = uploadData.downloadCount || 0;
    this.lastAccessed = uploadData.lastAccessed;
    this.expiresAt = uploadData.expiresAt || null;
    this.checksum = uploadData.checksum;
    this.createdAt = uploadData.createdAt;
    this.updatedAt = uploadData.updatedAt;
  }

  // Virtual properties
  get sizeFormatted() {
    const bytes = this.size;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  get extension() {
    return this.originalName.split('.').pop().toLowerCase();
  }

  get ageFormatted() {
    const now = new Date();
    const uploaded = new Date(this.createdAt);
    const diffTime = Math.abs(now - uploaded);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 1) return '1 day ago';
    if (diffDays < 30) return `${diffDays} days ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  }

  // Static methods
  static async findById(id) {
    const uploadData = await dataService.getUploadById(id);
    return uploadData ? new Upload(uploadData) : null;
  }

  static async findByUserId(userId) {
    const uploadsData = await dataService.getUploadsByUserId(userId);
    return uploadsData.map(uploadData => new Upload(uploadData));
  }

  static async findAll() {
    const uploadsData = await dataService.getUploads();
    return uploadsData.map(uploadData => new Upload(uploadData));
  }

  static async create(uploadData) {
    // Set file type based on mimetype
    if (uploadData.mimetype.startsWith('image/')) {
      uploadData.fileType = 'image';
    } else if (uploadData.mimetype.startsWith('video/')) {
      uploadData.fileType = 'video';
    }

    const newUploadData = await dataService.createUpload(uploadData);
    return new Upload(newUploadData);
  }

  // Instance methods
  async save() {
    const uploadData = {
      userId: this.userId,
      originalName: this.originalName,
      filename: this.filename,
      mimetype: this.mimetype,
      size: this.size,
      path: this.path,
      url: this.url,
      fileType: this.fileType,
      metadata: this.metadata,
      thumbnailPath: this.thumbnailPath,
      thumbnailUrl: this.thumbnailUrl,
      status: this.status,
      tags: this.tags,
      description: this.description,
      isPublic: this.isPublic,
      uploadSource: this.uploadSource,
      processingHistory: this.processingHistory,
      downloadCount: this.downloadCount,
      lastAccessed: this.lastAccessed,
      expiresAt: this.expiresAt,
      checksum: this.checksum
    };

    if (this.id) {
      const updatedData = await dataService.updateUpload(this.id, uploadData);
      if (updatedData) {
        Object.assign(this, updatedData);
      }
    } else {
      const newUploadData = await dataService.createUpload(uploadData);
      Object.assign(this, newUploadData);
    }
    
    return this;
  }

  async remove() {
    if (this.id) {
      return await dataService.deleteUpload(this.id);
    }
    return false;
  }

  async addProcessingHistory(action, details, userId) {
    this.processingHistory.push({
      action,
      details,
      user: userId,
      timestamp: new Date().toISOString()
    });
    return await this.save();
  }

  async incrementDownloadCount() {
    this.downloadCount += 1;
    this.lastAccessed = new Date().toISOString();
    return await this.save();
  }

  isExpired() {
    if (!this.expiresAt) return false;
    return new Date() > new Date(this.expiresAt);
  }
}

module.exports = Upload;
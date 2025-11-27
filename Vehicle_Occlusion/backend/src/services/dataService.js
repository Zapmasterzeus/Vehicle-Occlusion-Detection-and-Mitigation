const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

class DataService {
  constructor() {
    this.dataDir = path.join(__dirname, '../../data');
    this.usersFile = path.join(this.dataDir, 'users.json');
    this.uploadsFile = path.join(this.dataDir, 'uploads.json');
    this.detectionsFile = path.join(this.dataDir, 'detections.json');
    
    this.initializeDataFiles();
  }

  async initializeDataFiles() {
    try {
      // Ensure data directory exists
      await fs.ensureDir(this.dataDir);
      
      // Initialize files if they don't exist
      if (!await fs.pathExists(this.usersFile)) {
        await fs.writeJson(this.usersFile, []);
      }
      if (!await fs.pathExists(this.uploadsFile)) {
        await fs.writeJson(this.uploadsFile, []);
      }
      if (!await fs.pathExists(this.detectionsFile)) {
        await fs.writeJson(this.detectionsFile, []);
      }
    } catch (error) {
      console.error('Error initializing data files:', error);
    }
  }

  // Generic CRUD operations
  async readData(filename) {
    try {
      return await fs.readJson(filename);
    } catch (error) {
      console.error(`Error reading ${filename}:`, error);
      return [];
    }
  }

  async writeData(filename, data) {
    try {
      await fs.writeJson(filename, data, { spaces: 2 });
      return true;
    } catch (error) {
      console.error(`Error writing ${filename}:`, error);
      return false;
    }
  }

  // User operations
  async getUsers() {
    return await this.readData(this.usersFile);
  }

  async getUserById(id) {
    const users = await this.getUsers();
    return users.find(user => user.id === id);
  }

  async getUserByEmail(email) {
    const users = await this.getUsers();
    return users.find(user => user.email === email);
  }

  async createUser(userData) {
    const users = await this.getUsers();
    const newUser = {
      id: uuidv4(),
      ...userData,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    users.push(newUser);
    await this.writeData(this.usersFile, users);
    return newUser;
  }

  async updateUser(id, updateData) {
    const users = await this.getUsers();
    const userIndex = users.findIndex(user => user.id === id);
    if (userIndex === -1) return null;
    
    users[userIndex] = {
      ...users[userIndex],
      ...updateData,
      updatedAt: new Date().toISOString()
    };
    await this.writeData(this.usersFile, users);
    return users[userIndex];
  }

  async deleteUser(id) {
    const users = await this.getUsers();
    const filteredUsers = users.filter(user => user.id !== id);
    await this.writeData(this.usersFile, filteredUsers);
    return filteredUsers.length < users.length;
  }

  // Upload operations
  async getUploads() {
    return await this.readData(this.uploadsFile);
  }

  async getUploadById(id) {
    const uploads = await this.getUploads();
    return uploads.find(upload => upload.id === id);
  }

  async getUploadsByUserId(userId) {
    const uploads = await this.getUploads();
    return uploads.filter(upload => upload.userId === userId);
  }

  async createUpload(uploadData) {
    const uploads = await this.getUploads();
    const newUpload = {
      id: uuidv4(),
      ...uploadData,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    uploads.push(newUpload);
    await this.writeData(this.uploadsFile, uploads);
    return newUpload;
  }

  async updateUpload(id, updateData) {
    const uploads = await this.getUploads();
    const uploadIndex = uploads.findIndex(upload => upload.id === id);
    if (uploadIndex === -1) return null;
    
    uploads[uploadIndex] = {
      ...uploads[uploadIndex],
      ...updateData,
      updatedAt: new Date().toISOString()
    };
    await this.writeData(this.uploadsFile, uploads);
    return uploads[uploadIndex];
  }

  async deleteUpload(id) {
    const uploads = await this.getUploads();
    const filteredUploads = uploads.filter(upload => upload.id !== id);
    await this.writeData(this.uploadsFile, filteredUploads);
    return filteredUploads.length < uploads.length;
  }

  // Detection operations
  async getDetections() {
    return await this.readData(this.detectionsFile);
  }

  async getDetectionById(id) {
    const detections = await this.getDetections();
    return detections.find(detection => detection.id === id);
  }

  async getDetectionsByUserId(userId) {
    const detections = await this.getDetections();
    return detections.filter(detection => detection.userId === userId);
  }

  async getDetectionsByUploadId(uploadId) {
    const detections = await this.getDetections();
    return detections.filter(detection => detection.uploadId === uploadId);
  }

  async createDetection(detectionData) {
    const detections = await this.getDetections();
    const newDetection = {
      id: uuidv4(),
      ...detectionData,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    detections.push(newDetection);
    await this.writeData(this.detectionsFile, detections);
    return newDetection;
  }

  async updateDetection(id, updateData) {
    const detections = await this.getDetections();
    const detectionIndex = detections.findIndex(detection => detection.id === id);
    if (detectionIndex === -1) return null;
    
    detections[detectionIndex] = {
      ...detections[detectionIndex],
      ...updateData,
      updatedAt: new Date().toISOString()
    };
    await this.writeData(this.detectionsFile, detections);
    return detections[detectionIndex];
  }

  async deleteDetection(id) {
    const detections = await this.getDetections();
    const filteredDetections = detections.filter(detection => detection.id !== id);
    await this.writeData(this.detectionsFile, filteredDetections);
    return filteredDetections.length < detections.length;
  }

  // Statistics and analytics
  async getStats(userId = null) {
    const uploads = await this.getUploads();
    const detections = await this.getDetections();

    const userUploads = userId ? uploads.filter(u => u.userId === userId) : uploads;
    const userDetections = userId ? detections.filter(d => d.userId === userId) : detections;

    return {
      totalUploads: userUploads.length,
      totalDetections: userDetections.length,
      recentActivity: userUploads.slice(-10).reverse(),
      vehicleStats: this.calculateVehicleStats(userDetections),
      occlusionStats: this.calculateOcclusionStats(userDetections)
    };
  }

  calculateVehicleStats(detections) {
    const vehicleTypes = {};
    detections.forEach(detection => {
      if (detection.vehicles) {
        detection.vehicles.forEach(vehicle => {
          vehicleTypes[vehicle.type] = (vehicleTypes[vehicle.type] || 0) + 1;
        });
      }
    });
    return vehicleTypes;
  }

  calculateOcclusionStats(detections) {
    let totalVehicles = 0;
    let occludedVehicles = 0;
    
    detections.forEach(detection => {
      if (detection.vehicles) {
        detection.vehicles.forEach(vehicle => {
          totalVehicles++;
          if (vehicle.occluded) {
            occludedVehicles++;
          }
        });
      }
    });

    return {
      totalVehicles,
      occludedVehicles,
      occlusionRate: totalVehicles > 0 ? (occludedVehicles / totalVehicles * 100).toFixed(2) : 0
    };
  }
}

module.exports = new DataService();
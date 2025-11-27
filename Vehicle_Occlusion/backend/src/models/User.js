const bcrypt = require('bcryptjs');
const dataService = require('../services/dataService');

class User {
  constructor(userData) {
    this.id = userData.id;
    this.username = userData.username;
    this.email = userData.email;
    this.password = userData.password; // Should be hashed
    this.firstName = userData.firstName;
    this.lastName = userData.lastName;
    this.role = userData.role || 'user';
    this.avatar = userData.avatar || null;
    this.isActive = userData.isActive !== undefined ? userData.isActive : true;
    this.lastLogin = userData.lastLogin;
    this.preferences = userData.preferences || {
      theme: 'light',
      notifications: { email: true, push: true },
      language: 'en'
    };
    this.stats = userData.stats || {
      totalUploads: 0,
      totalDetections: 0,
      totalProcessingTime: 0,
      lastUploadDate: null
    };
    this.createdAt = userData.createdAt;
    this.updatedAt = userData.updatedAt;
  }

  // Virtual for full name
  get fullName() {
    return `${this.firstName} ${this.lastName}`;
  }

  // Static methods for database operations
  static async findById(id) {
    const userData = await dataService.getUserById(id);
    return userData ? new User(userData) : null;
  }

  static async findByEmail(email) {
    const userData = await dataService.getUserByEmail(email);
    return userData ? new User(userData) : null;
  }

  static async findByUsername(username) {
    const users = await dataService.getUsers();
    const userData = users.find(user => user.username === username);
    return userData ? new User(userData) : null;
  }

  static async create(userData) {
    // Hash password before saving
    if (userData.password) {
      const salt = await bcrypt.genSalt(12);
      userData.password = await bcrypt.hash(userData.password, salt);
    }
    
    const newUserData = await dataService.createUser(userData);
    return new User(newUserData);
  }

  static async findAll() {
    const usersData = await dataService.getUsers();
    return usersData.map(userData => new User(userData));
  }

  // Instance methods
  async save() {
    const userData = {
      username: this.username,
      email: this.email,
      password: this.password,
      firstName: this.firstName,
      lastName: this.lastName,
      role: this.role,
      avatar: this.avatar,
      isActive: this.isActive,
      lastLogin: this.lastLogin,
      preferences: this.preferences,
      stats: this.stats
    };

    if (this.id) {
      // Update existing user
      const updatedData = await dataService.updateUser(this.id, userData);
      if (updatedData) {
        Object.assign(this, updatedData);
      }
    } else {
      // Create new user
      if (this.password) {
        const salt = await bcrypt.genSalt(12);
        userData.password = await bcrypt.hash(userData.password, salt);
      }
      const newUserData = await dataService.createUser(userData);
      Object.assign(this, newUserData);
    }
    
    return this;
  }

  async remove() {
    if (this.id) {
      return await dataService.deleteUser(this.id);
    }
    return false;
  }

  async comparePassword(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
  }

  async updateLastLogin() {
    this.lastLogin = new Date().toISOString();
    return await this.save();
  }

  async incrementUploadStats() {
    this.stats.totalUploads += 1;
    this.stats.lastUploadDate = new Date().toISOString();
    return await this.save();
  }

  toJSON() {
    const userObject = { ...this };
    delete userObject.password;
    return userObject;
  }

  // Validation methods
  static validateRegistration(userData) {
    const errors = [];

    if (!userData.username || userData.username.length < 3) {
      errors.push('Username must be at least 3 characters long');
    }

    if (!userData.email || !this.isValidEmail(userData.email)) {
      errors.push('Valid email is required');
    }

    if (!userData.password || userData.password.length < 6) {
      errors.push('Password must be at least 6 characters long');
    }

    if (!userData.firstName || userData.firstName.length < 1) {
      errors.push('First name is required');
    }

    if (!userData.lastName || userData.lastName.length < 1) {
      errors.push('Last name is required');
    }

    return errors;
  }

  static isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }
}

module.exports = User;
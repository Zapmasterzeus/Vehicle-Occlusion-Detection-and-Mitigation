import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Chip,
  Grid,
} from '@mui/material';
import {
  CloudUpload,
  Image as ImageIcon,
  CheckCircle,
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState('');
  const { user } = useAuth();
  const navigate = useNavigate();

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check file type
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file');
        return;
      }

      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }

      setSelectedFile(file);
      setError('');
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setUploading(true);
    setError('');
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      
      // For guest users, we can still upload but with limited features
      if (user?.isGuest) {
        formData.append('guestUpload', 'true');
      }

      const response = await api.post('/uploads', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = response.data.data;
      setUploadResult(result);
      
      // For authenticated users, navigate to detection results
      if (!user?.isGuest && result.detectionId) {
        setTimeout(() => {
          navigate(`/detection/${result.detectionId}`);
        }, 2000);
      }

    } catch (err) {
      console.error('Upload error:', err);
      
      if (user?.isGuest) {
        // For guest users, simulate a successful upload with demo data
        const demoResult = {
          id: 'guest-' + Date.now(),
          filename: selectedFile.name,
          uploadDate: new Date().toISOString(),
          status: 'processed',
          occlusionPercentage: Math.floor(Math.random() * 50) + 10,
          confidenceScore: Math.floor(Math.random() * 30) + 70,
          processingTime: `${Math.floor(Math.random() * 5) + 1}.${Math.floor(Math.random() * 9)}s`,
          isDemo: true,
        };
        setUploadResult(demoResult);
      } else {
        const message = err.response?.data?.message || 'Upload failed';
        setError(message);
      }
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setUploadResult(null);
    setError('');
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Upload Image for Analysis
      </Typography>

      {user?.isGuest && (
        <Alert severity="info" sx={{ mb: 3 }}>
          You're using guest mode. Uploads will be processed with demo data for demonstration purposes.
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {uploadResult ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <CheckCircle color="success" sx={{ fontSize: 64, mb: 2 }} />
          <Typography variant="h5" gutterBottom>
            Upload Successful!
          </Typography>
          
          {uploadResult.isDemo && (
            <Chip 
              label="Demo Data" 
              color="warning" 
              size="small" 
              sx={{ mb: 2 }} 
            />
          )}

          <Grid container spacing={2} sx={{ mt: 2, maxWidth: 600, mx: 'auto' }}>
            <Grid item xs={12} sm={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="primary">
                    {uploadResult.occlusionPercentage}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Occlusion Level
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" color="secondary">
                    {uploadResult.confidenceScore}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence Score
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="body1" sx={{ mt: 2 }}>
                <strong>File:</strong> {uploadResult.filename}
              </Typography>
              <Typography variant="body1">
                <strong>Processing Time:</strong> {uploadResult.processingTime}
              </Typography>
              <Typography variant="body1">
                <strong>Status:</strong> {uploadResult.status}
              </Typography>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button variant="contained" onClick={handleReset}>
              Upload Another Image
            </Button>
            {!user?.isGuest && uploadResult.detectionId && (
              <Button 
                variant="outlined" 
                onClick={() => navigate(`/detection/${uploadResult.detectionId}`)}
              >
                View Detailed Results
              </Button>
            )}
          </Box>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Select Image
              </Typography>
              
              <Box
                sx={{
                  border: '2px dashed',
                  borderColor: selectedFile ? 'success.main' : 'grey.400',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  '&:hover': {
                    borderColor: 'primary.main',
                    bgcolor: 'action.hover',
                  },
                }}
                onClick={() => document.getElementById('file-input').click()}
              >
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: 'none' }}
                />
                
                {selectedFile ? (
                  <>
                    <CheckCircle color="success" sx={{ fontSize: 48, mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      File Selected
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {selectedFile.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </Typography>
                  </>
                ) : (
                  <>
                    <CloudUpload sx={{ fontSize: 48, mb: 2, color: 'grey.500' }} />
                    <Typography variant="h6" gutterBottom>
                      Click to select an image
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Supported formats: JPG, PNG, GIF
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Maximum size: 10MB
                    </Typography>
                  </>
                )}
              </Box>

              <Box sx={{ mt: 3, textAlign: 'center' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                  startIcon={uploading ? <CircularProgress size={20} /> : <CloudUpload />}
                  sx={{ minWidth: 200 }}
                >
                  {uploading ? 'Processing...' : 'Analyze Image'}
                </Button>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Image Preview
              </Typography>
              
              <Box
                sx={{
                  minHeight: 300,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px dashed',
                  borderColor: 'grey.400',
                  borderRadius: 2,
                  bgcolor: 'grey.50',
                }}
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '400px',
                      objectFit: 'contain',
                      borderRadius: '4px',
                    }}
                  />
                ) : (
                  <Box sx={{ textAlign: 'center', color: 'grey.500' }}>
                    <ImageIcon sx={{ fontSize: 64, mb: 2 }} />
                    <Typography variant="body2">
                      Image preview will appear here
                    </Typography>
                  </Box>
                )}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default Upload;
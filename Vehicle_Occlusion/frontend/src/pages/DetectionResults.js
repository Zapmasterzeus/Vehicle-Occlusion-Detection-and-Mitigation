import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

const DetectionResults = () => {
  const { id } = useParams();
  const [detection, setDetection] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useAuth();

  useEffect(() => {
    const fetchDetection = async () => {
      try {
        if (user?.isGuest) {
          setError('Guest users cannot view detection results');
          setLoading(false);
          return;
        }

        const response = await api.get(`/detections/${id}`);
        setDetection(response.data.data);
      } catch (err) {
        console.error('Error fetching detection:', err);
        setError(err.response?.data?.message || 'Failed to load detection results');
      } finally {
        setLoading(false);
      }
    };

    fetchDetection();
  }, [id, user]);

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <CircularProgress />
        <Typography sx={{ mt: 2 }}>Loading detection results...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  if (!detection) {
    return (
      <Alert severity="warning">
        Detection results not found
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Detection Results
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Analyzed Image
            </Typography>
            {detection.resultImage ? (
              <img
                src={detection.resultImage}
                alt="Detection Result"
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: '4px',
                }}
              />
            ) : (
              <Alert severity="info">
                Processed image not available
              </Alert>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Detection Status
                  </Typography>
                  <Chip
                    label={detection.status || 'Completed'}
                    color="success"
                    variant="outlined"
                  />
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Occlusion Level
                  </Typography>
                  <Typography variant="h4" color="primary">
                    {detection.occlusionPercentage || 0}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Confidence Score
                  </Typography>
                  <Typography variant="h4" color="secondary">
                    {detection.confidenceScore || 0}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Processing Time
                  </Typography>
                  <Typography variant="body1">
                    {detection.processingTime || 'N/A'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>

        {detection.metadata && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Additional Details
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {JSON.stringify(detection.metadata, null, 2)}
              </Typography>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default DetectionResults;